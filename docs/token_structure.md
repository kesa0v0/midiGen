
0. 설계 원칙 (명시)

LLM은 의도·구조·역할만 생성한다

연주자(REMI)는 실행만 담당한다

불안정성은 인터프리터에서 흡수한다

확장은 토큰 추가만으로 가능해야 한다

1. 문서 전체 구조

[GLOBAL]

[SECTION]+

GLOBAL 블록: 정확히 1개

SECTION 블록: 1개 이상

SECTION 순서 = 재생 순서

2. GLOBAL 블록 (Level 1: Core)

2.1 필수 토큰

[BPM:128]

[KEY:C_MAJOR]

[TIME_SIG:4/4]

허용 값

토큰 값

BPM 정수 40–240

KEY {C,Db,D,Eb,E,F,Gb,G,Ab,A,Bb,B}_{MAJOR,MINOR}

TIME_SIG 4/4, 3/4, 6/8

GLOBAL 토큰은 문서 최상단에 위치

중복 ❌

3. SECTION 블록 (핵심 구조)

3.1 기본 형태

[SECTION:VERSE]

[BARS:8]

[PROG: vi IV I V | vi IV I V]

3.2 SECTION 타입 (폐쇄 집합)

SECTION ∈ {

  INTRO,

  VERSE,

  CHORUS,

  BRIDGE,

  CLIMAX,

  OUTRO

}

변형 이름(VERSE_A 등) ❌

반복은 연주 단계에서 처리

4. PROG — Chord Progression

4.1 코드 토큰

CHORD ∈ { I, ii, iii, IV, V, vi, vii° }

4.2 Secondary Dominant

V/I, V/ii, V/iii, V/IV, V/V, V/vi

4.3 문법 (비트 해상도 강화) _ (Underscore)를 사용하여 코드가 유지됨을 표시하거나, 코드 개수로 등분.

Simple: [PROG: I vi IV V] (Bar 당 1개로 해석 or 균등 분할)

Split: [PROG: I_IV ii_V] (한 마디 안에서 반씩 쪼개짐)

Inversion: I/3 (3음 베이스), V/7 (7음 베이스)

5. Control Tokens (Level 2)

선택 사항, 누락되어도 정상 작동

5.1 DYNAMIC

[DYNAMIC:LOW]

[DYNAMIC:MID]

[DYNAMIC:HIGH]

[DYNAMIC:FADE_OUT]

5.2 MOVEMENT (Pitch bias)

[MOVEMENT:ASCENDING]

[MOVEMENT:DESCENDING]

[MOVEMENT:STATIC]

5.3 DENSITY (Note density)

[DENSITY:SPARSE]

[DENSITY:NORMAL]

[DENSITY:DENSE]

5.4 FEEL (Rhythmic feel)

[FEEL:STRAIGHT]

[FEEL:SWING]

[FEEL:SYNCOPATED]

5.5 ARTICULATION (연주 주법 힌트)

[ART:LEGATO] (부드럽게 이어짐)

[ART:STACCATO] (짧고 끊어서)

5.6 TRANSITION (섹션 끝 처리)

[FILL:YES] (섹션 마지막 마디에 필인 추가)

6. INSTRUMENT ROLE (Level 2.5)

악기 “선택”이 아니라 “역할 선언”

6.1 GLOBAL Instrument Roles (선택)

[INSTRUMENTS]

  MELODY: SYNTH

  HARMONY: PAD

  BASS: SUB

  DRUMS: STANDARD

전체 곡 기본값

SECTION에서 override 가능

6.2 SECTION Instrument Roles (선택)

[SECTION:CLIMAX]

[INSTRUMENTS]

  MELODY: STRINGS

  HARMONY: STRINGS

  BASS: ELECTRIC

6.3 Role 정의 및 허용 값

MELODY (주선율)

MELODY ∈ { SYNTH, PIANO, GUITAR, STRINGS, FLUTE }

HARMONY (코드/패드)

HARMONY ∈ { PAD, STRINGS, SYNTH_PAD, PIANO, ORGAN }

BASS

BASS ∈ { SUB, ELECTRIC, ACOUSTIC }

DRUMS

DRUMS ∈ { STANDARD, ELECTRONIC, MINIMAL }

6.4 설계 규칙

역할 누락 → 인터프리터 default

알 수 없는 값 → ignore + log

구체 악기명 ❌ (GM, SoundFont 의존 제거)

7. Annotation Tokens (Level 3)

v1.1 인터프리터는 무조건 무시

[STYLE_HINT:lofi hip hop]

[EMOTION:hopeful but restrained]

[REFERENCE:jrpg town theme]

[LOOP_HINT:seamless]

자유 텍스트 허용

학습 시 “존재만” 노출

8. 전체 예시 (v1.1)

[BPM:128]

[KEY:C_MAJOR]

[TIME_SIG:4/4]

[INSTRUMENTS]

  MELODY: SYNTH

  HARMONY: PAD

  BASS: SUB

  DRUMS: ELECTRONIC

[SECTION:INTRO]

[BARS:4]

[PROG: I I IV IV]

[DYNAMIC:LOW]

[DENSITY:SPARSE]

[SECTION:VERSE]

[BARS:8]

[PROG: vi IV I V | vi IV I V]

[DYNAMIC:MID]

[MOVEMENT:DESCENDING]

[SECTION:CLIMAX]

[BARS:8]

[PROG: IV V vi III | IV V I I]

[DYNAMIC:HIGH]

[MOVEMENT:ASCENDING]

[DENSITY:DENSE]

[INSTRUMENTS]

  MELODY: STRINGS

  HARMONY: STRINGS

  BASS: ELECTRIC

[SECTION:OUTRO]

[BARS:4]

[PROG: I IV I V/V]

[DYNAMIC:FADE_OUT]

[LOOP_HINT:seamless]

9. 인터프리터 필수 정책

9.1 Reject (에러)

GLOBAL 필수 토큰 누락

BARS ≠ PROG bar 수

허용되지 않은 CHORD

SECTION 타입 오류

9.2 Auto-fix / Ignore

Control 누락 → default

INSTRUMENT 누락 → default

Annotation → ignore

```plain
[BPM:120]
[KEY:Ab_MAJOR]
[TIME_SIG:4/4]

[INSTRUMENTS]
  MELODY: PIANO
  HARMONY: STRINGS
  BASS: SUB
  DRUMS: STANDARD

[SECTION:VERSE]
[BARS:4]
[PROG: I V/vii vi V | IV I/iii ii V]  <-- 전위(Slash) 코드 사용으로 베이스 라인 하행 유도
[DYNAMIC:MID]
[DENSITY:NORMAL]
[FEEL:STRAIGHT]

[SECTION:PRE_CHORUS]
[BARS:4]
[PROG: vi iii IV I | vi iii IV V]
[DYNAMIC:RISING]                     <-- MID, HIGH 대신 점진적 상승(구현 가능하다면)
[FILL:YES]                           <-- 다음 섹션(Chorus) 진입 전 드럼 필인 요청

[SECTION:CHORUS]
[BARS:8]
[PROG: I V vi IV | I V vi IV]
[DYNAMIC:HIGH]
[DENSITY:DENSE]
[INSTRUMENTS]
  MELODY: SYNTH_LEAD                 <-- 코러스에서 악기 변경
  HARMONY: SYNTH_PAD
  BASS: SYNTH_BASS
```
