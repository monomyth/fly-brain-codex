# Project effort estimate

For the reel we use **over 300 million processed tokens, mostly cached context, and over 30 logged hours**.

These are rough log-based figures for this Codex task, not a bill, unique written text, human labor, or total compute across all machines. The saved September 12 snapshot reported approximately 341.2 million cumulative tokens: 339.4 million input (330.6 million cached input) and 1.76 million output. Cached input is a subset of input; reasoning output is a subset of output. Neither is added a second time.

The time figure sums the `duration_ms` values of 48 completed task turns: about 31.59 hours, including tools and waits. It excludes unfinished turns and separate GPU training compute. Approximately 76.6 calendar hours elapsed between the first event and the snapshot; that is not active work time. Other agents/apps and Grok sessions are outside this estimate.

We use conservative rounded lower-bound wording because session counters can be rewound after recovery and this task is still accumulating work. Raw private conversation logs are not published. [Saved aggregate](evidence/effort.json).
