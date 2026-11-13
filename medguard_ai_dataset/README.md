# MedGuard AI synthetic dataset

All compounds, protocols, studies, counts, and recommendations in this package are fictional and created solely for software-engineering and RAG evaluation exercises. They are not medical facts and must not be used for diagnosis, prescribing, treatment, or patient care.

## Files
- `documents.jsonl`: 30 synthetic source documents with IDs, types, versions, dates, titles, and text.
- `test_questions.jsonl`: 25 evaluation questions, answerability labels, relevant document IDs, and expected answer points.

## Suggested use
1. Ingest `documents.jsonl` and preserve document metadata.
2. Chunk the `text` field while retaining `document_id`, `version`, and `date`.
3. Build dense retrieval and BM25 retrieval.
4. Fuse rankings with RRF, then rerank.
5. Grade evidence sufficiency against the original question.
6. For insufficient evidence, rewrite the search query and retry with a bounded loop.
7. Generate answers only from evidence; verify claims and cite document IDs.
8. Evaluate retrieval and answer quality against `test_questions.jsonl`.

## Important evaluation note
The `relevant_document_ids` field is a small gold set for this exercise. Some questions intentionally require abstention or careful handling of outdated versions and evidence limitations.
