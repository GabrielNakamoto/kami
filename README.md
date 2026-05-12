# chessformer

A pedagogical implementation of a deep learning chess model.

- Will explore AlphaZero style self play but starting with simple supervised Transformer policy network.
- Using [lichess elite](https://database.nikonoel.fr/) database for training.


## Mk1

### Notable techniques/improvements:
- Mark illegal moves as very negative before computing cross entropy loss [^1]
  - Discourages model from considering illegal moves
- Explotation of spatial symmetry:
  - Rotate board so models pieces are always facing them
  - Removes the model needing to learn relation between global current player and rotation of board tensor
- 8 ply history

### Input Representation:
- Board Tensor (8,64)
  - 8 ply history of 64 squares
  - Each square contains class 0->12 (piece per color)
- Global Tensor (7)
  - Current player
  - Half Move Count
  - Repetition Count
  - Model Kingside Castle Rights
  - Model Queenside Castle Rights
  - Opponent Kingside Castle Rights
  - Opponent Queenside Castle Rights
 
### Policy Representation:
- Alpha zero style Action Tensor (73) [^2]
 
### Model:
- Vanilla Transformer Blocks
  - RMSNorm, pre normalization
  - 0.1 Dropout on attn + ffn outputs
  - SILU for ffn activation
- 64 class position embedding
- 8 class ply embedding
- 13 class piece embedding
- Global tensor projected to model dim through FFN with silu activation
- Policy head projects to action dim (73)

[^1]:https://lczero.org/
[^2]:Silver D, Schrittwieser J, Simonyan K, Antonoglou I, Huang A, Guez A, Hubert T, Baker L, Lai M, Bolton A, Chen Y, Lillicrap T, Hui F, Sifre L, van den Driessche G, Graepel T, Hassabis D. Mastering the game of Go without human knowledge. Nature. 2017 Oct 18;550(7676):354-359. doi: 10.1038/nature24270. PMID: 29052630.
