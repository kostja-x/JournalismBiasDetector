# Early (Unsuccessful) Attempts and Lessons Learned

Before finalizing the current approach using local models and well-structured data, there were several earlier attempts to tackle complex semantic multi-label classification tasks. Unfortunately, these initial trials were time-consuming, yielded limited insights, and highlighted the need for better preparation and a more efficient methodology.

## Context

The initial idea was to rely on a semantic multi-label classification approach directly on raw or insufficiently prepared data. This approach attempted to identify subtle manipulative techniques in news articles without an adequately large, balanced, and curated dataset. Instead of producing meaningful outcomes, it resulted in extremely long training times, low performance scores, and difficulties in interpreting the results.

## Issues Encountered

- **Excessive Training Duration**:  
  During some runs, code segments took an excessively long time. As shown in one of the screenshots, a loop that attempted to print redundant paraphrase pairs ran for over 1743 minutes (nearly 29 hours), making even basic debugging and evaluation impractical.
  
- **Low Performance Scores**:  
  Even after long training periods, the precision, recall, and F1-scores remained disappointingly low. The classification reports demonstrate that the model could not effectively distinguish between manipulative categories with the given data and approach.
  
- **Inadequate Data Quality and Preparation**:  
  Without properly prepared training sets and targeted data augmentation, the model struggled to learn meaningful patterns. The attempts to run multi-label classification on complex semantic data “out of the box” did not provide usable results.

## Visual Evidence

You can find images illustrating these challenges in `docs/attachments/previous_attempts`. For example:

- **"Laufzeit Augmentation"**: A screenshot showing how long it took just to run through augmentation steps.
- **"Multilabeling_r Augmentation"**: Demonstrates the complexity and confusion in attempts to handle multi-label classification without proper data preparation.
- **"Prelabeling Prozess"**: Shows an early pre-labeling attempt that ended up being ineffective.
- **"Redundanzprüfung_Augmentation"**: Highlights a near-endless check for redundant paraphrases, running for days without yielding practical results.

These images underscore that the initial code and logic were not just slow, but stalled the entire experimental cycle, preventing timely adjustments.

## EVAL JSON Data

Additionally, evaluation data (JSON files containing experimental metrics and outcomes) are stored in `docs/attachments/EVAL_Data`. These files offer a deeper insight into the model’s performance metrics (or lack thereof) and serve as tangible evidence of why a strategic pivot was necessary.

## Conclusion

These early experiments taught valuable lessons:

1. **Importance of Quality Data**: Without well-curated, labeled, and sufficiently large datasets, even the most sophisticated multi-label approaches will underperform.

2. **Need for Efficient Approaches**: Spending days waiting for a single epoch to complete is unsustainable. A more practical approach with local models and incremental steps is essential.

3. **Iterative Refinement**: Before committing to a complex pipeline, it’s crucial to test simpler solutions and ensure that each step contributes to better results.

By documenting these prior, less successful attempts, the project narrative remains transparent and emphasizes the need for strategic data handling and method selection. This ensures that future work builds on these hard-earned lessons to avoid repeating the same pitfalls.