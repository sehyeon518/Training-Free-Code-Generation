PROBLEM_WITH_EXPERIENCE_TEMPLATE = """Provide a {language} solution for the following competitive programming question:
{problem}

When writing code, you MUST first carefully read and understand the helpful instructions and experiences:
{experiences}"""


EXECUTION_FEEDBACK_TEMPLATE = """
Your previous code failed one or more execution tests. The following issues were detected:

{feedback_block}

Fix your code and submit a new {language} solution.
Your response must again consist ONLY of {language} code enclosed in:
```{language}
...
```
Use the backticks for code only.
"""

FEEDBACK_LINE_TEMPLATE = """
- input: {input}
  expected_output: {expected_output}
  observed_output: {observed_output}
  error_type: {error_type}
  error_detail: {error_msg}
"""


SINGLE_ROLLOUT_SUMMARY_TEMPLATE = """An agent system may be provided with some experiences, and then it produces the following trajectory to solve the given problem. Please summarize the trajectory step-by-step:

1. Describe the action taken at each step, and whether any experience influenced that decision.
2. Identify bugs, logic errors, missing edge-case handling, formatting mistakes, timeouts, or infinite loops.
3. Preserve all core events of each step, regardless of correctness.

<trajectory>
{trajectory}
</trajectory>

<grading_result>
{grade}
</grading_result>

Only return the trajectory summary of each step, e.g.,
1. what happened in the first step and the core outcomes
2. what happened in the second step and the core outcomes
3. ..."""


SINGLE_QUERY_CRITIQUE_TEMPLATE = """An agent system is provided with a set of experiences and has tried to generate the code multiple times with both successful and wrong solutions. Review these code-generating attempt and extract generalizable experiences. Follow these steps:

1. Trajectory Analysis:
    - For successful steps: Identify key correct programming strategies.
    - For errors: Pinpoint incorrect logic, failing patterns, and runtime issues.
    - Note any important patterns or strategies used/missed
    - Detect repeated failure modes across rollouts.
    - Review why some trajectories fail. Is there any existing experiences are missed, or experiences do not provide enough guidance?

2. Update Existing Experiences
    - Some trajectories may be correct and others may be wrong, you should ensure there are experiences can help to run correctly
    - You have two options: [modify, add]
        * modify: You can modify current experiences to make it helpful
        * add: You can introduce new experiences may need to be 
    - You can update at most {max_operations} clear, generalizable lessons for this case
    - Before updating each experience, you need to:
        * Specify when it would be most relevant
        * List key problem features that make this experience applicable
        * Identify similar problem patterns where this advice applies
    
3. Requirements for each experience that is modified or added.
    - Begin with general background with several words in the experience
    - Focus on strategy, debugging, and structural decisions
    - Emphasize decision points that could apply to similar problems

Please provide reasoning in details under the guidance of the above 3 steps.
After the step-by-step reasoning, you will finish by returning in this JSON format as follows:
```json
[
    {{
        "option": "modify",
        "experience": "the modified experience",
        "modified_from": "G17" # specify the ID of experience that is modified
    }},
    {{
        "option": "add",
        "experience": "the added experience",
    }},
    ...
]
```
Note that your updated experiences may not need to cover all two options. Only using one type of updates is also very good.

<problem> 
{problem}
</problem>

<trajectories>
{trajectories}
</trajectories>

<experiences>
{experiences}
</experiences>"""


BATCH_EXPERIENCE_UPDATE_TEMPLATE = """An agent system is provided with a set of experiences and has tried to generate the code multiple times. From the reflections, some suggestions on the existing experiences have been posed. Your task is to collect and think for the final experience revision plan. Each final experience must satisfy the following requirements.
1. It must be clear, generalizable lessons for this case, with no more than 32 words
2. Begin with general background with several words in the experience
3. Focus on strategy, debugging, and structural decisions
4. Emphasize decision points that could apply to similar problems
5. Avoid repeating saying similar experience in multiple different experiences

<existing_experiences> 
{experiences}
</existing_experiences>

<suggested_updates>
{updates}
</suggested_updates>

Please provide reasoning in each of the suggestions, and think for how to update existing experiences 
You have two update options: [modify, merge]
* modify: You can modify current experiences to make it helpful
* merge: You can merge some similar experiences into a more general forms to reduce duplication

After generating the step-by-step reasoning, you need to give the final experience revision details by returning in this JSON format as follows:
```json
[
    {{
        "option": "modify",
        "experience": "the modified experience",
        "modified_from": "C1" # specify the str ID of experience that is modified
    }},
    {{
        "option": "merge",
        "experience": "the merged experience",
        "merged_from": ["C1", "C3", "S4", ...] # specify the str IDs of experiences that is merged from, at least 2 IDs are needed
    }},
    ...
]
```

Your updated experiences may not need to cover all two options. Only using one type of updates is OK."""