import argparse
import subprocess
import os

def midi_to_wav(midi_path, wav_path, soundfont_path):
    cmd = [
        "fluidsynth", "-ni", soundfont_path, midi_path, "-F", wav_path, "-r", "44100"
    ]
    subprocess.run(cmd, check=True)

def wav_to_mp3(wav_path, mp3_path):
    cmd = [
        "ffmpeg", "-y", "-i", wav_path, mp3_path
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="입력 MIDI 파일 경로")
    parser.add_argument("-o", "--output", help="출력 MP3 파일 경로 (없으면 입력과 동일)")
    parser.add_argument("-s", "--soundfont", 
                        default="./data/soundfonts/Arachno SoundFont - Version 1.0.sf2",
                        help="SoundFont(.sf2) 파일 경로 (기본값: ./data/soundfonts/Arachno SoundFont - Version 1.0.sf2)"
                        )
    args = parser.parse_args()

    midi_path = args.input
    base = os.path.splitext(os.path.basename(midi_path))[0]
    wav_path = base + ".wav"
    mp3_path = args.output if args.output else base + ".mp3"

    midi_to_wav(midi_path, wav_path, args.soundfont)
    wav_to_mp3(wav_path, mp3_path)
    os.remove(wav_path)  # WAV 파일 삭제(원하면 주석처리)

    print(f"변환 완료: {mp3_path}")