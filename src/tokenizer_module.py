import logging
from pathlib import Path
import omegaconf
import numpy as np
from miditoolkit import MidiFile as ToolkitMidiFile 
from midiutil import MIDIFile
import os

# anticipation 라이브러리 임포트 (설치 필요)
try:
    from anticipation.convert import events_to_midi, midi_to_events
    _ANTICIPATION_AVAILABLE = True
except ImportError:
    log = logging.getLogger(__name__)
    log.warning("Anticipation library not found. AnticipationTokenizerWrapper will use dummy functions.")
    _ANTICIPATION_AVAILABLE = False

log = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import miditok

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
        self.config = config
        
        # --- 1. Special Tokens 설정 (0~99번 예약) ---
        self._pad_token_id = 0
        self._bos_token_id = 1
        self._eos_token_id = 2
        self._unk_token_id = 3
        self._mask_token_id = 4
        
        # 커스텀 메타데이터 토큰 (앞서 논의한 Context/Composer)
        self.special_token_offset = 100 

        # --- 2. AMT Vocabulary 설정 ---
        # Based on snippet: AMT_GPT2_BOS_ID = 55026
        # This implies the valid token range for AMT is roughly 0 to 55025.
        self._base_amt_vocab_size = 55026
        self.amt_vocab_start = self.special_token_offset 
        self._vocab_size = self.amt_vocab_start + self._base_amt_vocab_size

        log.info(f"Initialized Anticipation Tokenizer.")
        log.info(f"Total Vocab Size: {self._vocab_size} (Special: {self.special_token_offset} + AMT: {self._base_amt_vocab_size})")

    def encode(self, midi_path: Path) -> list:
        """MIDI -> Events -> Tokens"""
        if not midi_path.exists():
            log.error(f"MIDI file not found: {midi_path}")
            return []

        if not _ANTICIPATION_AVAILABLE:
            log.warning(f"Anticipation library not available. Returning dummy tokens for {midi_path.name}")
            return [self.start_token_id, 101, 102, 103, self.bar_token_id, 104, self.eos_token_id, self.pad_token_id] * 5 # Dummy

        try:
            # 1. MIDI -> Events (Tokens)
            # midi_to_events returns a list of integer tokens
            events = midi_to_events(str(midi_path))
            
            # 2. Shift tokens by offset
            amt_tokens = [t + self.amt_vocab_start for t in events]

            # 3. Add BOS/EOS
            return [self._bos_token_id] + amt_tokens + [self._eos_token_id]

        except Exception as e:
            log.error(f"Failed to encode {midi_path}: {e}")
            return []

    def decode(self, tokens: list, output_path: Path):
        """Tokens -> Events -> MIDI"""
        if not _ANTICIPATION_AVAILABLE:
            log.warning(f"Anticipation library not available. Generating dummy MIDI for {output_path.name}")
            midi = MIDIFile(1)
            track = 0
            time = 0
            midi.addTrackName(track, time, "Anticipation Dummy Track")
            midi.addTempo(track, time, 120)
            # Add some dummy notes based on tokens, if tokens are within a reasonable range
            for i, token in enumerate(tokens):
                if 100 <= token < 228: # Example: map some AMT tokens to note numbers (offset for pitch)
                    midi.addNote(track, track, token - self.amt_vocab_start, time + i * 0.25, 0.5, 100)
                elif token == self.bar_token_id: # Simulate a bar line
                    time += 4 # Move time forward for a new bar
            
            os.makedirs(output_path.parent, exist_ok=True)
            with open(output_path, "wb") as output_file:
                midi.writeFile(output_file)
            return


        try:
            # 1. Filter and Shift back
            valid_tokens = [
                t - self.amt_vocab_start
                for t in tokens
                if self.amt_vocab_start <= t < (self.amt_vocab_start + self._base_amt_vocab_size)
            ]
            
            if not valid_tokens:
                log.warning("No valid AMT tokens found to decode.")
                return

            # 2. Tokens -> MIDI
            # events_to_midi returns a miditoolkit/mido object with a .save method
            midi_obj = events_to_midi(valid_tokens)
            midi_obj.save(str(output_path))
            
            log.info(f"Saved decoded MIDI to {output_path}")

        except Exception as e:
            log.error(f"Failed to decode to {output_path}: {e}")

    # --- Property ---
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id
    
    @property
    def start_token_id(self) -> int:
        return self._bos_token_id
        
    @property
    def bar_token_id(self) -> int:
        # Placeholder: Anticipation might not have a direct "bar" token.
        return self.config.get('bar_token_id', 50) 


    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

def get_tokenizer(cfg: omegaconf.DictConfig) -> BaseTokenizer:
    tokenizer_name = cfg.tokenizer.name
    if tokenizer_name == "remi":
        # Pass the remi specific config
        return RemiTokenizerWrapper(cfg.tokenizer.remi)
    elif tokenizer_name == "anticipation":
        # Pass the anticipation specific config.
        return AnticipationTokenizerWrapper(cfg.tokenizer.anticipation)
    else:
        raise ValueError(f"Unknown tokenizer name: {tokenizer_name}")