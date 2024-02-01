from torch import nn
from transformers import BertConfig, BertModel


class DialogTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_config = BertConfig(
            num_hidden_layers=6,
            vocab_size=21128,
            hidden_size=512,
            num_attention_heads=8,

        )
        self.encoder = BertModel(encoder_config)

        decoder_config = BertConfig(
            num_hidden_layers=6,
            vocab_size=21128,
            hidden_size=512,
            num_attention_heads=8,
            add_cross_attention=True,
        )
        decoder_config.is_decoder = True
        self.decoder = BertModel(decoder_config)

        self.linear = nn.Linear(512, 21128, bias=False)

    def forward(self, input_ids, mask_encoder_input, output_ids, mask_decoder_input):
        outputs = self.encoder(input_ids, mask_encoder_input)
        encoder_hidden_states = outputs.last_hidden_state
        # out: [batch_size, max_length, hidden_size]
        out = self.decoder(output_ids, mask_decoder_input, encoder_hidden_states=encoder_hidden_states)
        decoder_hidden_states = out.last_hidden_state
        out = self.linear(decoder_hidden_states)
        return out