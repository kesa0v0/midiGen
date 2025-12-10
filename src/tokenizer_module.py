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
        
        # --- 1. Basic Special Tokens (0-9) ---
        self._pad_token_id = 0
        self._bos_token_id = 1
        self._eos_token_id = 2
        self._unk_token_id = 3
        self._mask_token_id = 4
        
        # --- 2. Structural & Semantic Special Tokens (10-99) ---
        # Structure Headers
        self.tokens_structure = {
            'Global_Header': 10,
            'Memory_Anchors': 11,
            'Narrative_Stream': 12,
        }
        
        # Metadata keys/values (Simplified placeholders)
        self.tokens_meta = {
            'Genre_Fantasy_Orchestral': 20,
            'Composer_Hans_Zimmer_Style': 21,
            'Composer_Chopin': 22,
            'BPM_Variable': 23,
        }
        
        # Anchor Definitions
        self.tokens_anchor = {
            'Define_Motif_A': 30,
            'Define_Motif_B': 31,
            'End_Motif': 32,
        }
        
        # Narrative Contexts
        self.tokens_context = {
            'Context_Peaceful_Village': 40,
            'Context_Battle': 41,
            'Context_Tragic_Loss': 42,
            'Event_Surprise_Attack': 43,
            'Intensity_Low': 44,
            'Intensity_High': 45,
            'Intensity_Medium': 46,
        }

        # Offset for AMT tokens (0-99 reserved for special tokens)
        self.special_token_offset = 100 

        # --- 3. AMT Vocabulary Setting ---
        # Based on snippet: AMT_GPT2_BOS_ID = 55026
        # This implies the valid token range for AMT is roughly 0 to 55025.
        self._base_amt_vocab_size = 55026
        self.amt_vocab_start = self.special_token_offset 
        
        # Calculate raw vocab size
        raw_vocab_size = self.amt_vocab_start + self._base_amt_vocab_size
        
        # Optimization: Pad vocab size to be a multiple of 128 (Tensor Core friendly)
        # 55126 -> 55168 (adds ~42 dummy tokens)
        alignment = 128
        self._vocab_size = ((raw_vocab_size + alignment - 1) // alignment) * alignment
        
        log.info(f"Initialized Anticipation Tokenizer.")
        log.info(f"Raw Vocab Size: {raw_vocab_size} -> Aligned Vocab Size: {self._vocab_size} (Multiple of {alignment})")

    def _generate_dummy_metadata_tokens(self) -> list:
        """Generates a structured header with dummy metadata."""
        # [Global_Header]
        #    [Genre: Fantasy_Orchestral]
        #    [Composer: Hans_Zimmer_Style] (Dummy, normally inferred)
        #    [BPM: Variable]
        header = [
            self.tokens_structure['Global_Header'],
            self.tokens_meta['Genre_Fantasy_Orchestral'],
            self.tokens_meta['Composer_Hans_Zimmer_Style'],
            self.tokens_meta['BPM_Variable']
        ]
        
        # [Memory_Anchors] (Dummy empty definitions for now)
        #    [Define_Motif_A] ... [End_Motif]
        # In future, we can extract main themes from the midi and put them here
        anchors = [
            self.tokens_structure['Memory_Anchors'],
            self.tokens_anchor['Define_Motif_A'],
            self.tokens_anchor['End_Motif']
        ]
        
        return header + anchors

    def encode(self, midi_path: Path) -> list:
        """MIDI -> Structured Tokens (Header + Anchors + Stream)"""
        # Ensure midi_path is a Path object
        if isinstance(midi_path, str):
            midi_path = Path(midi_path)

        if not midi_path.exists():
            log.error(f"MIDI file not found: {midi_path}")
            return []

        if not _ANTICIPATION_AVAILABLE:
            log.warning(f"Anticipation library not available. Returning dummy tokens for {midi_path.name}")
            return [self.start_token_id, 101, 102, 103, self.bar_token_id, 104, self.eos_token_id, self.pad_token_id] * 5 

        try:
            # 1. MIDI -> AMT Events (Tokens)
            events = midi_to_events(str(midi_path))
            # Shift tokens by offset
            amt_tokens = [t + self.amt_vocab_start for t in events]
            
            # 2. Construct Structured Sequence
            # [BOS]
            # [Global_Header] ... [Memory_Anchors] ... (Dummy Meta)
            # [Narrative_Stream]
            #    [Context: Peaceful_Village] (Default Context)
            #    [MIDI_Tokens...]
            # [EOS]
            
            structure_prefix = self._generate_dummy_metadata_tokens()
            stream_start = [
                self.tokens_structure['Narrative_Stream'],
                self.tokens_context['Context_Peaceful_Village'], # Default context for now
                self.tokens_context['Intensity_Medium']
            ]
            
            return [self._bos_token_id] + structure_prefix + stream_start + amt_tokens + [self._eos_token_id]

        except Exception as e:
            log.error(f"Failed to encode {midi_path}: {e}")
            return []

    def decode(self, tokens: list, output_path: Path):
        """Tokens -> Filter Structural Tokens -> Events -> MIDI"""
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
            # 1. Filter out all Special Tokens (0-99) and shift back AMT tokens
            valid_tokens = []
            for t in tokens:
                if self.amt_vocab_start <= t < (self.amt_vocab_start + self._base_amt_vocab_size):
                    valid_tokens.append(t - self.amt_vocab_start)
            
            if not valid_tokens:
                log.warning("No valid AMT tokens found to decode.")
                return

            # 2. Tokens -> MIDI
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