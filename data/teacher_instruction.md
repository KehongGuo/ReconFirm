# Teacher instruction (rung 1: corrections of the verifier's wrong answers)

Frozen 2026-10-09 before any correction was written. Teacher: Claude Opus 5.5 (`claude-opus-5-5`), run once as a
subagent of the experiment session; its output file is the artifact the training reads (hash recorded per round).

You are correcting a vision model, the *verifier* of a robot search team. It was shown an image crop and asked the
question in `prompt`; its answer (`student_answer`) was wrong. The ground truth (`truth`) comes from the scene
annotation and is not up for debate: your job is to write the answer the verifier should have given, not to decide
what the object is.

For each job line (one JSON object per line in the jobs file):

1. Open the image at `image` and look at the object in the centre.
2. Choose `category`: one entry of `vocabulary`, spelled exactly as listed.
   - If `truth` says the centre object **is** the target, `category` is the target.
   - If `truth` says it is **not** the target, `category` must not be the target. Use the annotation names in
     `truth` to pick the closest vocabulary entry (e.g. "sitting bench" -> bench, "tv stand" -> cabinet,
     "flower vase" -> plant only if the target is not plant, otherwise the nearest other entry such as furniture).
     If nothing fits, use `none`.
3. `clear`: true only if the crop is sharp and large enough that a person could name the centre object from it.
4. `description`: one sentence, in the style of the student's description, that names the object and states the
   visible feature that distinguishes it from what it was mistaken for (e.g. "dry twigs in a vase, no leaves",
   "a framed print on the wall, flat with a visible frame edge, not a screen"). Describe only what is visible.
5. `reflection`: true only if the centre object is a reflection in a mirror or glass.

Output one JSON object per job, in the same order, to the output file:
`{"id": ..., "category": ..., "clear": true|false, "description": "...", "reflection": true|false, "author": "teacher"}`

Do not skip jobs. Do not add commentary to the output file. If an image cannot be opened, write the line with
`"category": null` and the reason in `description`.

Amendment 2026-10-09, after batch r1_a, before any training: the answer must also follow the rules written in
`prompt` itself. A reflection is `none` with `reflection: true`; a depicted object is `picture`; an object not in
the list (e.g. a vase when the target is plant) is `none`. Batch r1_a was corrected by this rule (15 lines, marked
`amended`); the earlier example "flower vase -> furniture" was removed.

## Audit mode (rungs 2 and 3)

For an audit jobs file, each line also holds the student's own correction (`student_correction`). Output
`{"id": ..., "agree": true|false, "note": "..."}`: agree when the student's category satisfies the truth rule above
and its description is accurate for the image and names a distinguishing feature.
