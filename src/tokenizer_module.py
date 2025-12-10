# src/tokenizer_module.py

from abc import ABC, abstractmethod
import miditok
import omegaconf
import logging
from pathlib import Path
# Assuming 'anticipation' can be imported after installation
import anticipation # noqa: F401 (ignore flake8 unused import warning for now)
from midiutil import MIDIFile # For dummy MIDI generation in placeholder

log = logging.getLogger(__name__)

class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, midi_path: Path) -> list:
        pass

    @abstractmethod
    def decode(self, tokens: list, output_path: Path):
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        pass

    @property
    @abstractmethod
    def start_token_id(self) -> int:
        pass
    
    @property
    @abstractmethod
    def bar_token_id(self) -> int:
        pass

class RemiTokenizerWrapper(BaseTokenizer):
    def __init__(self, config: omegaconf.DictConfig):
        # Dynamically evaluate beat_res
        beat_res = eval(config.beat_res) if isinstance(config.beat_res, str) else config.beat_res
        
        # miditok expects nb_velocities, our config uses num_velocities
        nb_velocities = config.nb_velocities if hasattr(config, 'nb_velocities') else 32
        
        additional_tokens_config = {
            'Chord': config.get('use_chords', False),
            'Rest': config.get('use_rests', False),
            'Program': config.get('use_programs', False),
            'Tempo': config.get('use_tempos', False),
            'time_signature': config.get('use_time_signatures', False),
            'rest_range': eval(config.get('rest_range', '[2, 8]')) if isinstance(config.get('rest_range'), str) else config.get('rest_range', [2, 8]),
            'tempo_range': eval(config.get('tempo_range', '[50, 200]')) if isinstance(config.get('tempo_range'), str) else config.get('tempo_range', [50, 200]),
            'ProgramChanges': config.get('use_program_changes', False)
        }
        
        self.tokenizer = miditok.REMI(
            beat_res=beat_res,
            nb_velocities=nb_velocities,
            additional_tokens=additional_tokens_config
        )
        log.info(f"Initialized REMI Tokenizer with vocab size: {self.tokenizer.vocab_size}")
        # Ensure [START] token is added and recognized
        if '[START]' not in self.tokenizer._token_to_id:
            self.tokenizer.add_special_tokens(['[START]'])
            log.info(f"REMI Tokenizer after adding [START] token, vocab size: {self.tokenizer.vocab_size}")

    def encode(self, midi_path: Path) -> list:
        # miditok returns a list of tokens, each having an 'ids' attribute
        tokens = self.tokenizer(str(midi_path))
        if tokens and hasattr(tokens[0], 'ids'):
            return tokens[0].ids
        elif tokens: # Older miditok versions might return list of ints directly
            return tokens
        return []

    def decode(self, tokens: list, output_path: Path):
        # miditok.decode expects a list of token sequences
        midi_object = self.tokenizer.decode([tokens])
        midi_object.dump_midi(str(output_path))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer['PAD_None'] # REMI's default PAD token

    @property
    def start_token_id(self) -> int:
        return self.tokenizer['[START]']

    @property
    def bar_token_id(self) -> int:
        return self.tokenizer['Bar_None']


class AnticipationTokenizerWrapper(BaseTokenizer):
    def __init__(self, config: omegaconf.DictConfig):
        # In a real scenario, you'd load/initialize the anticipation tokenizer here.
        # This might involve defining an event vocabulary or loading a pre-trained one.
        log.warning("AnticipationTokenizerWrapper is a placeholder and requires actual implementation.")
        self.config = config
        
        # These would typically come from the anticipation library's vocabulary
        self._vocab_size = config.get('vocab_size', 1000) # Example default
        self._pad_token_id = config.get('pad_token_id', 0) # Example default
        self._start_token_id = config.get('start_token_id', 1) # Example default
        self._bar_token_id = config.get('bar_token_id', 2) # Example default, assuming 'bar' concept exists
        log.info(f"Initialized Anticipation Tokenizer (Placeholder) with vocab size: {self._vocab_size}")

    def encode(self, midi_path: Path) -> list:
        # TODO: Implement encoding for anticipation tokenizer
        # This would involve:
        # 1. Loading the MIDI file: midi_data = anticipation.load_midi(midi_path)
        # 2. Converting MIDI data to anticipation's event representation: events = anticipation.midi_to_events(midi_data)
        # 3. Converting events to numerical tokens: tokens = anticipation.events_to_tokens(events, self.vocabulary)
        log.warning(f"AnticipationTokenizerWrapper encode is a placeholder. Encoding {midi_path.name}")
        # Return dummy tokens for now
        return [self.start_token_id, 3, 4, 5, self.bar_token_id, 6, 7, self.pad_token_id] * 10

    def decode(self, tokens: list, output_path: Path):
        # TODO: Implement decoding for anticipation tokenizer
        # This would involve:
        # 1. Converting numerical tokens back to anticipation's event representation: events = anticipation.tokens_to_events(tokens, self.vocabulary)
        # 2. Converting events to MIDI data: midi_data = anticipation.events_to_midi(events)
        # 3. Saving the MIDI data: anticipation.save_midi(midi_data, output_path)
        log.warning(f"AnticipationTokenizerWrapper decode is a placeholder. Decoding to {output_path.name}")
        
        # Create a dummy MIDI file for now using midiutil
        midi = MIDIFile(1)
        track = 0
        time = 0
        midi.addTrackName(track, time, "Anticipation Dummy Track")
        midi.addTempo(track, time, 120)
        # Add some dummy notes based on tokens, if tokens are within a reasonable range
        for i, token in enumerate(tokens):
            if 60 <= token < 72: # Example: map some tokens to note numbers
                midi.addNote(track, time + i * 0.5, token, 1, 0.5, 100) # Note, channel, pitch, time, duration, velocity
            elif token == self.bar_token_id: # Simulate a bar line
                time += 4 # Move time forward for a new bar
        
        with open(output_path, "wb") as output_file:
            midi.writeFile(output_file)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def start_token_id(self) -> int:
        return self._start_token_id

    @property
    def bar_token_id(self) -> int:
        return self._bar_token_id

def get_tokenizer(cfg: omegaconf.DictConfig) -> BaseTokenizer:
    tokenizer_name = cfg.tokenizer.name
    if tokenizer_name == "remi":
        # Pass the remi specific config
        return RemiTokenizerWrapper(cfg.tokenizer.remi)
    elif tokenizer_name == "anticipation":
        # Pass the anticipation specific config, along with global max_seq_len and vocab_size if needed
        # For a more robust solution, 'anticipation' config would have its own vocab_size etc.
        anticipation_cfg_with_globals = omegaconf.OmegaConf.merge(cfg.tokenizer.anticipation, 
                                                                  omegaconf.OmegaConf.create({'vocab_size': cfg.tokenizer.vocab_size}))
        return AnticipationTokenizerWrapper(anticipation_cfg_with_globals)
    else:
        raise ValueError(f"Unknown tokenizer name: {tokenizer_name}")

