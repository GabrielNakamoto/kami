# kami

> A deep learning chess model prioritizing elo/parameter ratio and convergence speed for consumer GPUs.

## Data
Ended up deriving a custom dataset from the [lichess elite](https://database.nikonoel.fr/) database. I enrich and reduce the original dataset by picking n sample positions (n=3 currently) at random from each game, excluding the first 6 and last position to reduce opening memorization. This results in decorrelated positions, intended to allow for faster model generalization and reduce reliance on temporal patterns without true 'understanding'. The position is stored in FEN format with the next move label in UCI. For value head training, we obtain a stockfish WDL (win/draw/loss) estimate for the position and normalize to \[0,1\]. Stockfish is used to distill a richer training signal then just scalar game outcomes. 

I encoded the dataset in parquet file format for efficient storage and retrieval and uploaded it to hugging face for public usage [here](https://huggingface.co/datasets/gRa1ne/decorrelated-chess).

The dataset processing scripts are in the `data/` directory, however there are some shell commands necessary to download and build a raw PGN file to reproduce the dataset (I plan to write a bash script in the future). 

Before training the model you need to run `data/process.py` to expand the dataset into tensors with the proper formatting. This is multiprocessed using numpy memmaps to be as efficient as possible.

## Input Representation
Game state/input is encoded as 2 tensors:
- a 64 element piece tensor with classes 0-12 representing every possible piece configuration per square. Not one hot encoded as it is fed through an embedding layer that projects to dim.
- A 9 element global feature tensor broadcasted as a bias across every square. Contains all other useful information about game state such as: castling rights per side, en pessant squares, repetition counts and the half-move clock count.

## Output Representation
Inspired by leela. 1858 element tensor constructed from a mapping of every possible movement between two squares including underpromotions. That is an integer index for each (from square, to square, promotion type or null) that is legal.

## Model
Essentially a transformer network with an input embedding block and policy head output.

Notable design choices and domain improvements:
- Leela 'chessformer'/smolgen architecture, implemented as an opptional attention flag. Improves attention with relative position encoding, heuristically describing the semantic relation between positions through chess moves rather than euclidian distance.[^1][^2]
- Muon optimizer on 2d matrices to speed up convergence[^3]
- Mixed precision linear layers and activation functions for massive tensor core gains[^4]

[^1]: https://lczero.org/blog/2024/02/transformer-progress/
[^2]: https://arxiv.org/abs/2409.12272
[^3]: https://arxiv.org/pdf/2502.16982
[^4]: https://arxiv.org/pdf/1905.12322
