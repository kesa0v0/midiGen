```plain
[BOS]
[Global_Header]      # 1. 세션 설정 (전역 컨텍스트)
   [Genre: Fantasy_Orchestral]
   [Composer: Hans_Zimmer_Style]
   [BPM: Variable]

[Memory_Anchors]     # 2. 기억해야 할 핵심 테마 (Titan 메모리 적재용)
   [Define_Motif_A] [MIDI_Seq...] [End_Motif]  # 메인 테마 (용사의 주제)
   [Define_Motif_B] [MIDI_Seq...] [End_Motif]  # 서브 테마 (비극적 사랑)

[Narrative_Stream]   # 3. 실제 서사 진행 (상황 변화에 따른 변주 훈련)
   [Context: Peaceful_Village] [Intensity: Low]
      [MIDI_Seq (Motif_A_Calm_Variation)...]
   
   [Event: Surprise_Attack]     # 갑작스러운 상황 변화 토큰
   [Context: Battle] [Intensity: High]
      [MIDI_Seq (Motif_A_Epic_Variation)...]   # 앞서 정의된 Motif A가 격렬하게 변주됨
   
   [Context: Tragic_Loss] [Intensity: Medium]
      [MIDI_Seq (Motif_B_Slow_Variation)...]   # Motif B 등장
[EOS]
```