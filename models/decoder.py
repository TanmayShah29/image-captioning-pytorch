"""
LSTM Decoder for Image Captioning
===================================
Generates captions word-by-word from image features.

Supports:
  • Teacher-forced forward pass (training)
  • Greedy decoding (fast inference)
  • Beam search decoding (higher quality)
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """LSTM-based caption decoder.

    Parameters
    ----------
    embed_size  : int  – word-embedding & encoder-feature dimension
    hidden_size : int  – LSTM hidden state size
    vocab_size  : int  – number of words in vocabulary
    num_layers  : int  – stacked LSTM layers  (default 1)
    dropout     : float – dropout probability  (default 0.5)
    """

    def __init__(self, embed_size, hidden_size, vocab_size,
                 num_layers=1, dropout=0.5):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self._init_weights()

        print(f"Decoder  embed={embed_size}  hidden={hidden_size}  "
              f"vocab={vocab_size}  layers={num_layers}")

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()

    # ------------------------------------------------------------------
    # Training forward (teacher forcing)
    # ------------------------------------------------------------------

    def forward(self, features, captions):
        """
        Args:
            features : (B, embed_size)   – from encoder
            captions : (B, max_length)   – target word indices

        Returns:
            outputs  : (B, max_length-1, vocab_size)
        """
        # Remove last token — we don't predict *after* <end>
        captions = captions[:, :-1]
        embeddings = self.embed(captions)                       # (B, L-1, E)
        features = features.unsqueeze(1)                        # (B, 1,   E)
        lstm_in = torch.cat((features, embeddings), dim=1)      # (B, L,   E)
        lstm_out, _ = self.lstm(lstm_in)                        # (B, L,   H)
        lstm_out = self.dropout(lstm_out)
        outputs = self.linear(lstm_out)                         # (B, L,   V)
        # Position 0 corresponds to image features; predictions start at 1
        return outputs[:, 1:, :]                                # (B, L-1, V)

    # ------------------------------------------------------------------
    # Greedy decoding
    # ------------------------------------------------------------------

    def generate_caption(self, features, vocab, max_length=50):
        """Greedy decode a single image.

        Args:
            features : (1, embed_size)
            vocab    : Vocabulary
            max_length : int

        Returns:
            list of str — caption words (without special tokens)
        """
        caption = []
        hidden = None

        # First step: [image_features, <start>]
        start_tok = torch.tensor([[vocab.word2idx["<start>"]]],
                                 device=features.device)
        start_emb = self.embed(start_tok)                        # (1,1,E)
        lstm_in = torch.cat((features.unsqueeze(1), start_emb), dim=1)   # (1,2,E)
        lstm_out, hidden = self.lstm(lstm_in, hidden)
        scores = self.linear(lstm_out[:, -1, :])                 # (1,V)
        pred = scores.argmax(dim=1)

        word = vocab.idx2word.get(pred.item(), "<unk>")
        if word == "<end>":
            return caption
        if word not in ("<start>", "<pad>", "<unk>"):
            caption.append(word)

        inp = pred.unsqueeze(1)
        for _ in range(max_length - 1):
            emb = self.embed(inp)                                # (1,1,E)
            lstm_out, hidden = self.lstm(emb, hidden)
            scores = self.linear(lstm_out[:, -1, :])
            pred = scores.argmax(dim=1)
            word = vocab.idx2word.get(pred.item(), "<unk>")
            if word == "<end>":
                break
            if word not in ("<start>", "<pad>"):
                caption.append(word)
            inp = pred.unsqueeze(1)

        return caption

    # ------------------------------------------------------------------
    # Beam search decoding
    # ------------------------------------------------------------------

    def beam_search(self, features, vocab, beam_size=3, max_length=50):
        """Beam-search decode a single image.

        Args:
            features  : (1, embed_size)
            vocab     : Vocabulary
            beam_size : int
            max_length: int

        Returns:
            list of str — best caption words
        """
        device = features.device
        end_idx = vocab.word2idx["<end>"]
        start_idx = vocab.word2idx["<start>"]

        # Initial step
        start_tok = torch.tensor([[start_idx]], device=device)
        start_emb = self.embed(start_tok)
        lstm_in = torch.cat((features.unsqueeze(1), start_emb), dim=1)
        lstm_out, hidden = self.lstm(lstm_in)
        scores = self.linear(lstm_out[:, -1, :])                  # (1, V)
        log_probs = torch.log_softmax(scores, dim=-1)             # (1, V)

        # Top-k starting words
        topk_lp, topk_idx = log_probs.topk(beam_size, dim=-1)    # (1, k)

        # beams: list of (log_prob, [word_indices], hidden_state)
        beams = []
        for i in range(beam_size):
            h = (hidden[0].clone(), hidden[1].clone())
            beams.append((topk_lp[0, i].item(), [topk_idx[0, i].item()], h))

        completed = []

        for _ in range(max_length - 1):
            candidates = []
            for lp, seq, hid in beams:
                last = seq[-1]
                if last == end_idx:
                    completed.append((lp, seq))
                    continue
                inp = torch.tensor([[last]], device=device)
                emb = self.embed(inp)
                out, new_hid = self.lstm(emb, hid)
                sc = self.linear(out[:, -1, :])
                lps = torch.log_softmax(sc, dim=-1)
                topk_lps, topk_ids = lps.topk(beam_size, dim=-1)
                for j in range(beam_size):
                    new_lp = lp + topk_lps[0, j].item()
                    new_seq = seq + [topk_ids[0, j].item()]
                    new_h = (new_hid[0].clone(), new_hid[1].clone())
                    candidates.append((new_lp, new_seq, new_h))

            if not candidates:
                break

            # Length-normalised score
            candidates.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
            beams = candidates[:beam_size]

        # Add remaining beams to completed
        for lp, seq, _ in beams:
            completed.append((lp, seq))

        if not completed:
            return []

        # Best by length-normalised log-prob
        best_seq = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))[1]

        # Convert to words, filtering special tokens
        words = []
        for idx in best_seq:
            w = vocab.idx2word.get(idx, "<unk>")
            if w == "<end>":
                break
            if w not in ("<start>", "<pad>"):
                words.append(w)
        return words


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    E, H, V, B, L = 256, 512, 5000, 4, 20
    dec = Decoder(E, H, V)
    feats = torch.randn(B, E)
    caps = torch.randint(0, V, (B, L))
    out = dec(feats, caps)
    print(f"\nForward  in: feats={feats.shape}  caps={caps.shape}")
    print(f"Forward out: {out.shape}  (expected ({B}, {L-1}, {V}))")
    params = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    print(f"Trainable params: {params:,}")
    print("✓ Decoder test passed")
