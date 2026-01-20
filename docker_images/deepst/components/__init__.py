from .registry import (
    get_encoder,
    get_graph_encoder,
    get_decoder,
    register_encoder,
    register_graph_encoder,
    register_decoder,
)

from .encoder_deepst import Encoder
from .graph_encoder_deepst import GraphEncoder
from .decoder_deepst import Decoder, InnerProductDecoder

# Import _other templates to register them
from . import encoder_other
from . import decoder_other
