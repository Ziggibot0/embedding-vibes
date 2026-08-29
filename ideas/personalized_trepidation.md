# Personalized Trepidation — per-user continual fine-tuning (product twist)

## Idea
The 8-dim A-space trunk trains once (MLM + Barlow Twins, format-invariant).
A small personalization head sits on top: user upvotes/downvotes their agent's
behavior → gradient update on the head only → the model learns what "trouble"
means FOR THIS USER on THIS USER's tasks.

## Why it's different from the base idea
The base monitor predicts generic success/failure. This predicts *user
satisfaction*, which is:
- task-specific (exp8 LOBO showed "trouble" is benchmark-local)
- user-specific (two users on the same task may want different things)
- not definable by a general benchmark — so the lexical bar problem
  (can a bag of words do this?) partly dissolves: TF-IDF can count words,
  but it can't learn one person's preference function from sparse votes.

## Why it's cheap
- Trunk: trained once, frozen at deploy. 6M params.
- Personalization head: linear layer (8 dims → 1), a few hundred params.
  Each upvote/downvote = one gradient step. Runs on-device, no GPU needed.
- No session logs leave the user's machine — only the gradient signal.
  Privacy is a selling point, not a cost.

## Why it might not matter if the trunk is lexical
Even if the 8-dim space is "just" a compressed lexicon at the trunk level, the
personalization head can learn which lexical patterns THIS user flags as bad.
The bar shifts from "beat TF-IDF on a general task" to "learn a user's
preference function faster than a TF-IDF+logreg on their votes." Sparse
per-user data (~10-50 votes) may actually favor the neural head if the trunk
already encodes useful structure from pretraining.

## Risks
- Cold start: before any votes, the head is random. Need a default
  (general trepidation or zero-signal until enough votes).
- Overfitting to few votes: 8 dims + 1 head is low-capacity, but 10 votes
  could still overfit. Regularize toward the general head.
- User gaming: adversarial users could feed bad votes. Low priority for v1.

## Un-park trigger
exp9 attempt #2 or #3 shows the trunk has signal (G1 passes). Then the
personalization head is a 1-day add-on: frozen trunk + per-user linear head +
online SGD. First test: does 20 simulated user votes improve AUC over the
general head on a held-out user's preferences?

## Relationship to the dissertation
This is the product chapter, not the science chapter. The science chapter
proves A-space exists; this chapter shows it's deployable as a personalized
product. Committees like product-grounded research — but only if the science
chapter is solid first.