import argparse
import asyncio
import copy
import json
import os
import random

from training_free_grpo.main import rollout_dataset, load_rollouts
from training_free_grpo.util_coex import *
from utu.agents import SimpleAgent
from utu.config import ConfigLoader

random.seed(42)


async def main(args):
    # Set up domain-specific variables
    if args.domain == "math":
        from training_free_grpo.math.dataset import load_data
        from training_free_grpo.math.verify import verify_func
        from training_free_grpo.math.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        from training_free_grpo.math.experience import ExperienceUpdater
        config_name = "simple/math_agent.yaml"
    elif args.domain == "web":
        from training_free_grpo.web.dataset import load_data
        from training_free_grpo.web.verify import verify_func
        from training_free_grpo.web.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        from training_free_grpo.web.experience import ExperienceUpdater
        config_name = "simple/base_search.yaml"
    elif args.domain == "code":
        from training_free_grpo.code.dataset import load_data
        from training_free_grpo.code.verify import verify_func
        from training_free_grpo.code.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        from training_free_grpo.code.experience import ExperienceUpdater
        config_name = "simple/code_agent.yaml"
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")
    
    # Create experiment directory
    experiment_dir = os.path.join("data", args.domain, "train", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Set up the agent
    if args.mode == "prompt":
        worker_agent = None
    elif args.mode == "agent":
        config = ConfigLoader.load_agent_config(config_name)
        config.model.model_settings.temperature = args.rollout_temperature
        worker_agent = SimpleAgent(config=config)
        await worker_agent.build()
    else:
        raise ValueError(f"Unsupported inference mode: {args.mode}")

    # Load the dataset
    train_data = load_data(args.dataset)
    print(f"Loaded {len(train_data)} records from dataset")
    if args.dataset_truncate is not None:
        print(f"- truncated to {args.dataset_truncate}")
        train_data = train_data[: args.dataset_truncate]
    # assert len(train_data) % args.batchsize == 0

    # Set up the stats
    stats_filename = os.path.join(experiment_dir, "stats.json")
    if os.path.exists(stats_filename):
        stats = json.load(open(stats_filename))
    else:
        stats = {}

    # Train
    for epoch in range(args.epochs):
        # Init
        print("=" * 30 + f"\nEpoch {epoch}\n" + "=" * 30)
        cur_epoch_dir = os.path.join(experiment_dir, f"epoch_{epoch}")
        os.makedirs(cur_epoch_dir, exist_ok=True)

        # Check if shuffled data already exists for this epoch
        shuffled_filename = os.path.join(cur_epoch_dir, "shuffled_data.jsonl")
        if os.path.exists(shuffled_filename):
            shuffled_data = []
            with open(shuffled_filename) as f:
                for line in f:
                    shuffled_data.append(json.loads(line))
            print(f"Loaded {len(shuffled_data)} records from shuffled data")
        else:
            print(f"Shuffling data ...")
            shuffled_data = copy.deepcopy(train_data)
            random.shuffle(shuffled_data)
            with open(shuffled_filename, "w") as f:
                for each in shuffled_data:
                    f.write(json.dumps(each) + "\n")

        # for each batch
        num_batches = len(shuffled_data) // args.batchsize
        for batch_idx in range(num_batches):
            step = epoch * num_batches + batch_idx
            if f"step_{step}" not in stats:
                stats[f"step_{step}"] = {"epoch": epoch, "batch": batch_idx, "complete": False}
            elif stats[f"step_{step}"]["complete"]:
                continue

            print(f"Step {step} (Epoch {epoch}, Batch {batch_idx})")
            cur_step_dir = os.path.join(experiment_dir, f"step_{step}")
            os.makedirs(cur_step_dir, exist_ok=True)

            batch_data = copy.deepcopy(
                shuffled_data[batch_idx * args.batchsize : (batch_idx + 1) * args.batchsize]
            )

            rollout_filename = os.path.join(cur_step_dir, "rollout.jsonl")
            rollouts = load_rollouts(rollout_filename)
            
            if step > 0:
                experience_filename = os.path.join(cur_step_dir, "experiences.json")
                experiences = json.load(open(experience_filename, encoding="utf-8"))

                prob_path = os.path.join(cur_step_dir, "experience_probs.json")
                experience_probs = load_and_init_experience_probs(experiences, prob_path)
            else:
                experiences = {}
                experience_probs = {}


            print(f"Loaded {len(experience_probs)} experience probabilities")

            formatted_batch_data = []
            for each in batch_data: # for each problem in the batch
                for rollout_idx in range(args.grpo_n):
                    if experiences:
                        if args.experience_sampling_mode == "uniform":
                            sampled_experiences = sample_experiences_uniform_k(experiences, k=args.num_experiences)
                        elif args.experience_sampling_mode == "grpo": # TODO: Seohee's new function
                            sampled_experiences = sample_experiences_grpo(experiences, experience_probs, k=args.num_experiences,rng=random)
                        else:
                            sampled_experiences = experiences
                        formatted_experiences = "\n".join(
                            [f"[{i}]. {e}" for i, e in sampled_experiences.items()]
                        )
                    else:
                        sampled_experiences = {}
                        formatted_experiences = "None"
                    
                    prompt = (
                        PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
                            experiences=formatted_experiences,
                            problem=each["problem"],
                            language=each.get("language", "python"),
                        ) if experiences
                        else each["problem"]
                    )

                    formatted_batch_data.append(
                        {
                            "prompt": prompt,
                            **each,
                            "sampled_experience_ids": list(sampled_experiences.keys()),
                        }
                    )
            # ============================================================
            
            # Duplicate for GRPO
            print(f"GRPO rollout number={args.grpo_n}")
            formatted_batch_data = formatted_batch_data * args.grpo_n if args.experience_sampling_mode=="query" else formatted_batch_data

            # Rollout the dataset
            rollouts, rollout_stats = await rollout_dataset(
                worker_agent=None,
                data=formatted_batch_data,
                rollouts=rollouts,
                verify_func=verify_func,
                rollout_filename=rollout_filename,
                rollout_concurrency=args.rollout_concurrency,
                task_timeout=args.task_timeout,
                temperature=args.rollout_temperature,
                max_tokens=args.rollout_max_tokens,
            )

            stats[f"step_{step}"]["rollout"] = rollout_stats

            # Generate critiques and update experiences
            next_step_dir = os.path.join(experiment_dir, f"step_{step+1}")
            os.makedirs(next_step_dir, exist_ok=True)
            next_experience_filename = os.path.join(next_step_dir, "experiences.json")
            if os.path.exists(next_experience_filename):
                print(f"Experiences already exist for step {step}, skipping experience update")
            else:
                new_experiences = ExperienceUpdater().run(
                    rollouts=rollouts, 
                    experiences=experiences,
                    save_dir=cur_step_dir,
                    max_workers=args.rollout_concurrency,
                    given_ground_truth=True if args.given_ground_truth=="True" else False,
                    only_partial_correct=True if args.grpo_n > 1 else False,
                )
                json.dump(new_experiences, open(next_experience_filename, "w"), indent=2)
                print(f"Saved {len(new_experiences)} experiences to {next_experience_filename}")

            # Extract "sampled_experience_ids" from partial_correct rollouts
            critique_path = os.path.join(cur_step_dir, "single_query_critique.json")
            problem_results = json.load(open(critique_path, encoding="utf-8"))

            batch_update_path = os.path.join(cur_step_dir, "batch_update.json")
            batch_update_result = json.load(open(batch_update_path, encoding="utf-8"))

            next_experiences = json.load(open(next_experience_filename, encoding="utf-8"))

            next_experience_probs_filename = os.path.join(next_step_dir, "experience_probs.json")
            debug_filename = os.path.join(cur_step_dir, "debug.json")
            _ = load_and_init_experience_probs(next_experiences, next_experience_probs_filename)


            new_experience_probs, debug_obj = compute_next_experience_probs(
                experiment_dir=experiment_dir,
                cur_step=step,
                cur_experiences=experiences,
                cur_probs=experience_probs,
                problem_results=problem_results,
                batch_update_result=batch_update_result,
                next_experiences=next_experiences,
            )

            with open(next_experience_probs_filename, "w") as f:
                json.dump(new_experience_probs, f, indent=2, ensure_ascii=False)

            with open(debug_filename, "w") as f:
                json.dump(debug_obj, f, indent=2, ensure_ascii=False)

            # Save stats
            stats[f"step_{step}"]["complete"] = True
            json.dump(stats, open(stats_filename, "w"), indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training-free GRPO")
    parser.add_argument("--mode", type=str, default="agent", required=True, choices=["prompt", "agent"], help="Mode of inference")
    parser.add_argument("--domain", type=str, required=True, choices=["math", "web", "code"], help="domain of the tasks (math/web/code)")
    parser.add_argument("--experiment_name", type=str, required=True, help="name of experiment run")
    parser.add_argument("--dataset", type=str, required="True", help="Name of dataset")
    parser.add_argument("--dataset_truncate", type=int, default=None, help="Truncate dataset to first N samples")
    parser.add_argument("--given_ground_truth", type=str, default="True", help="Whether use ground truth answers")
    parser.add_argument("--epochs", type=int, default=2, help="number of training epochs")
    parser.add_argument("--batchsize", type=int, default=64, help="batchsize")
    parser.add_argument("--grpo_n", type=int, default=5, help="number of rollouts in a group of GRPO")
    parser.add_argument("--rollout_concurrency", type=int, default=1, help="Concurrency level for rollouts")
    parser.add_argument("--rollout_temperature", type=float, default=0.7, help="Temperature for the LLM")
    parser.add_argument("--rollout_max_tokens", type=int, default=16384, help="Max tokens for each rollout batch")
    parser.add_argument("--task_timeout", type=float, default=60, help="Timeout for each individual task in seconds")
    parser.add_argument("--num_experiences", type=int, default=30, help="Number of experiences to maintain")
    parser.add_argument("--experience_sampling_mode", type=str, default=None, choices=["uniform", "grpo"])

    args = parser.parse_args()
    asyncio.run(main(args))