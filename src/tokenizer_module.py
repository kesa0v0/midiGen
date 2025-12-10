import logging
from pathlib import Path
import omegaconf
import numpy as np
import miditoolkit
from midiutil import MIDIFile
import os

# anticipation 라이브러리 임포트 (더 이상 직접 사용하지 않지만, 기존 코드를 정리하며 남겨둠)
try:
    from anticipation.convert import events_to_midi, midi_to_events
    _ANTICIPATION_AVAILABLE = True
except ImportError:
    log = logging.getLogger(__name__)
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
        log.debug(f"Initialized REMI Tokenizer with vocab size: {self.tokenizer.vocab_size}")
        # Ensure [START] token is added and recognized
        if '[START]' not in self.tokenizer._token_to_id:
            self.tokenizer.add_special_tokens(['[START]'])
            log.debug(f"REMI Tokenizer after adding [START] token, vocab size: {self.tokenizer.vocab_size}")

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
        
        # --- 2. Structural & Semantic Special Tokens (10-99) ---
        self.tokens_structure = {
            'Global_Header': 10, 'Memory_Anchors': 11, 'Narrative_Stream': 12, 'Time_Shift': 13 # Added Time_Shift
        }
        self.tokens_meta = {
            'Genre_Fantasy_Orchestral': 20, 'Composer_Hans_Zimmer_Style': 21,
            'Composer_Chopin': 22, 'BPM_Variable': 23,
        }
        self.tokens_anchor = {
            'Define_Motif_A': 30, 'Define_Motif_B': 31, 'End_Motif': 32,
        }
        self.tokens_context = {
            'Context_Peaceful_Village': 40, 'Context_Battle': 41, 'Context_Tragic_Loss': 42,
            'Event_Surprise_Attack': 43, 'Intensity_Low': 44, 'Intensity_High': 45, 'Intensity_Medium': 46,
        }

        # --- 3. Factorized AMT Vocabulary Definition ---
        self.special_token_offset = 100 
        
        # Ranges definition (Offset, Count)
        # 1. Onset: 0 ~ 100s (10ms unit) -> 10000 tokens
        self.vocab_onset = (self.special_token_offset, 10000) 
        self.TIME_SEGMENT_SIZE = 100.0 # seconds, for onset normalization
        
        # 2. Duration: 0 ~ 10s (10ms unit) -> 1000 tokens
        self.vocab_dur = (self.vocab_onset[0] + self.vocab_onset[1], 1000)
        
        # 3. Instrument: 0 ~ 128 (128 is Drum) -> 129 tokens
        self.vocab_inst = (self.vocab_dur[0] + self.vocab_dur[1], 129)
        
        # 4. Pitch: 0 ~ 127 -> 128 tokens
        self.vocab_pitch = (self.vocab_inst[0] + self.vocab_inst[1], 128)
        
        # 5. Velocity: 0 ~ 31 (Quantized 32 levels) -> 32 tokens
        self.vocab_vel = (self.vocab_pitch[0] + self.vocab_pitch[1], 32)
        
        # Calculate Total Vocab Size
        raw_vocab_size = self.vocab_vel[0] + self.vocab_vel[1]
        
        # Optimization: Alignment for Tensor Cores
        alignment = 128
        self._vocab_size = ((raw_vocab_size + alignment - 1) // alignment) * alignment
        
        log.info(f"Initialized Factorized AMT Tokenizer.")
        log.info(f"Structure: Onset->Dur->Inst->Pitch->Vel")
        log.info(f"Total Vocab Size: {self._vocab_size} (Optimized from ~300k)")

    # --- Helper: Value to Token ID ---
    def _val2tok(self, val, vocab_range, clip=True):
        start_idx, size = vocab_range
        if clip:
            val = max(0, min(val, size - 1))
        return start_idx + int(val)

    # --- Helper: Token ID to Value ---
    def _tok2val(self, token, vocab_range):
        start_idx, size = vocab_range
        if start_idx <= token < start_idx + size:
            return token - start_idx
        return None

    def _generate_dummy_metadata_tokens(self) -> list:
        """Generates a structured header with dummy metadata."""
        header = [
            self.tokens_structure['Global_Header'],
            self.tokens_meta['Genre_Fantasy_Orchestral'],
            self.tokens_meta['Composer_Hans_Zimmer_Style'],
            self.tokens_meta['BPM_Variable']
        ]
        anchors = [
            self.tokens_structure['Memory_Anchors'],
            self.tokens_anchor['Define_Motif_A'],
            self.tokens_anchor['End_Motif']
        ]
        return header + anchors

    def encode(self, midi_path: Path) -> list:
        """MIDI -> Factorized AMT Tokens"""
        if isinstance(midi_path, str): midi_path = Path(midi_path)
        if not midi_path.exists(): return []

        try:
            # 1. Load MIDI using miditoolkit (Not anticipation lib)
            midi_obj = miditoolkit.MidiFile(str(midi_path))
            
            notes_data = []
            # Gather all notes from all instruments
            for inst in midi_obj.instruments:
                for note in inst.notes:
                    ticks_per_beat = midi_obj.ticks_per_beat
                    
                    start_sec = (note.start / ticks_per_beat) * 0.5 
                    end_sec = (note.end / ticks_per_beat) * 0.5
                    
                    duration = end_sec - start_sec
                    
                    inst_idx = 128 if inst.is_drum else inst.program
                    vel_idx = int(note.velocity / 4)
                    
                    notes_data.append((start_sec, duration, inst_idx, note.pitch, vel_idx))
            
            # Sort by Onset time
            notes_data.sort(key=lambda x: x[0])
            
            # 2. Convert to Tokens with Time Segmentation
            amt_tokens = []
            current_time_segment = 0
            
            for note_start_sec, note_duration, inst_idx, pitch, vel_idx in notes_data:
                # Calculate which segment this note belongs to
                target_segment = int(note_start_sec // self.TIME_SEGMENT_SIZE)
                
                # Insert Time_Shift tokens if segments are skipped
                while current_time_segment < target_segment:
                    amt_tokens.append(self.tokens_structure['Time_Shift'])
                    current_time_segment += 1
                
                # Normalize onset within the current segment
                relative_onset_sec = note_start_sec % self.TIME_SEGMENT_SIZE
                
                onset_idx = int(relative_onset_sec * 100) # 10ms unit
                dur_idx = int(note_duration * 100)       # 10ms unit
                
                # Factorized: 5 tokens per note
                amt_tokens.append(self._val2tok(onset_idx, self.vocab_onset)) # Onset
                amt_tokens.append(self._val2tok(dur_idx, self.vocab_dur))     # Duration
                amt_tokens.append(self._val2tok(inst_idx, self.vocab_inst))   # Inst
                amt_tokens.append(self._val2tok(pitch, self.vocab_pitch))     # Pitch
                amt_tokens.append(self._val2tok(vel_idx, self.vocab_vel))     # Vel

            # 3. Add Structure
            structure_prefix = self._generate_dummy_metadata_tokens()
            stream_start = [
                self.tokens_structure['Narrative_Stream'],
                self.tokens_context['Context_Peaceful_Village'],
                self.tokens_context['Intensity_Medium']
            ]
            
            return [self._bos_token_id] + structure_prefix + stream_start + amt_tokens + [self._eos_token_id]

        except Exception as e:
            log.error(f"Encoding failed for {midi_path}: {e}")
            return []

    def decode(self, tokens: list, output_path: Path):
        """Factorized Tokens -> MIDI"""
        try:
            # 1. Filter structural tokens
            # Include Time_Shift token in valid_tokens filtering
            valid_tokens = [t for t in tokens if t >= self.special_token_offset or t == self.tokens_structure['Time_Shift']]
            
            # 2. Parse 5-gram structure
            notes = []
            current_segment_offset_sec = 0.0 # Tracks the total time offset from Time_Shift tokens
            buffer = {} # store parsed values
            
            for t in valid_tokens:
                if t == self.tokens_structure['Time_Shift']:
                    current_segment_offset_sec += self.TIME_SEGMENT_SIZE
                    continue # Skip to next token after handling Time_Shift
                    
                # Detect token type
                val_onset = self._tok2val(t, self.vocab_onset)
                val_dur = self._tok2val(t, self.vocab_dur)
                val_inst = self._tok2val(t, self.vocab_inst)
                val_pitch = self._tok2val(t, self.vocab_pitch)
                val_vel = self._tok2val(t, self.vocab_vel)
                
                if val_onset is not None:
                    # New note starts, use normalized onset
                    buffer = {'start_relative': val_onset / 100.0} # 10ms -> sec
                
                elif val_dur is not None and 'start_relative' in buffer:
                    buffer['duration'] = val_dur / 100.0
                
                elif val_inst is not None and 'duration' in buffer: # Changed from 'end' to 'duration'
                    buffer['inst'] = val_inst
                    
                elif val_pitch is not None and 'inst' in buffer:
                    buffer['pitch'] = val_pitch
                    
                elif val_vel is not None and 'pitch' in buffer:
                    buffer['vel'] = val_vel * 4 # De-quantize
                    
                    # Note Complete!
                    # Calculate actual start and end time with segment offset
                    actual_start_sec = buffer['start_relative'] + current_segment_offset_sec
                    actual_end_sec = actual_start_sec + buffer['duration']

                    # Ticks = Seconds * 960 (assuming 120 BPM, 480 TPB)
                    ticks_start = int(actual_start_sec * 960)
                    ticks_end = int(actual_end_sec * 960)
                    
                    new_note = miditoolkit.Note(
                        velocity=int(buffer['vel']),
                        pitch=int(buffer['pitch']),
                        start=ticks_start,
                        end=ticks_end
                    )
                    # Attach inst info for later grouping
                    new_note.inst_program = buffer['inst']
                    notes.append(new_note)
                    
                    buffer = {} # Reset

            # 3. Create MIDI Object
            midi_obj = miditoolkit.MidiFile()
            midi_obj.ticks_per_beat = 480
            
            # Group notes by instrument
            inst_map = {}
            for n in notes:
                prog = getattr(n, 'inst_program', 0)
                is_drum = (prog == 128)
                if is_drum: prog = 0 # Map drum to standard program 0 but on drum track
                
                key = (prog, is_drum)
                if key not in inst_map:
                    inst_map[key] = miditoolkit.Instrument(program=prog, is_drum=is_drum, name=f"Inst {prog}")
                
                inst_map[key].notes.append(inst) # ERROR: n instead of inst
            
            for inst in inst_map.values():
                midi_obj.instruments.append(inst)
            
            os.makedirs(output_path.parent, exist_ok=True)
            midi_obj.dump(str(output_path))
            log.info(f"Saved decoded MIDI to {output_path}")

        except Exception as e:
            log.error(f"Decoding failed for {output_path}: {e}")

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