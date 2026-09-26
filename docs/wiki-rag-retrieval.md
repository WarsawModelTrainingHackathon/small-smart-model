# Wikipedia retrieval check on the history exam dev split

This is a manual relevance review of the 42 `dev` items from the 2024 May R exam.
The retrieval code saw each item's question and source context only; it does not
read the answer, answer key, or scoring rubric. The test split was not opened.

## Results

| Cutoff | Items with at least one topic-relevant passage | Rate |
|---:|---:|---:|
| top 3 | 33 / 42 | 78.6% |
| top 5 | 34 / 42 | 81.0% |
| top 10 | 36 / 42 | 85.7% |

Relevance was judged by reviewing retrieved titles and passage text. A hit means
that at least one passage is directly about the requested historical topic or
provides useful context for it. This is a retrieval check, not an answer-accuracy
score. Source-comparison and image-interpretation questions can still require
evidence from the exam itself even when Wikipedia returns useful context.

The six items without a useful passage in the first ten were the question about
the November Uprising's name (only Chłopicki background ranked), a source-only
interpretation of Chłopicki's opinion, two Sedan/Hohenzollern questions, an
image-only cartoon interpretation, and the broad essay topic on Polish-Ottoman
wars. Some other hits were indirect: for example, the relevant Monroe Doctrine
article was rank 7, and Józef Poniatowski was rank 8.

## Latency

On box B, 20 warmed representative queries had 4.11 ms median, 9.37 ms p90, and
16.01 ms maximum search latency. The 42 longer exam queries took about 2.4 s
total (57 ms average) with the dev query builder and a larger result scan.
Cold-start latency is higher because Tantivy has to page index data in from disk.

## Index

The full Polish `20231101.pl` corpus produced 2,613,073 passages from 1,435,766
articles. The index is local and disk-based; the build used one indexing thread
and a bounded writer heap. The source dataset identifies the Wikimedia Wikipedia
corpus and its CC BY-SA 3.0 / GFDL licenses: <https://huggingface.co/datasets/wikimedia/wikipedia/tree/ec52a81a5f1fcd86daee3b05d6f39d2d6d849288/20231101.pl>.
