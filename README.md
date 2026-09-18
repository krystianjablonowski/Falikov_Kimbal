# FK Transport: DMFT/TMT dla nieuporządkowanego modelu Falicova–Kimballa

Niezależna implementacja obliczeń przewodności elektrycznej i cieplnej w
spinless modelu Falicova–Kimballa na sieci Bethego. Kod nie importuje ani nie
wykorzystuje starego `dmft_fkm.py`.

Pakiet realizuje dwie osobne gałęzie samozgodności:

- arytmetyczną DMFT, w której uśredniane jest zespolone lokalne `G`;
- TMT, w której typowa DOS jest rekonstruowana do `G` przez zeropaddingowaną
  transformację Hilberta FFT.

Samenergia jest zachowywana w surowej postaci. Kod nie obcina dodatniej części
`Im Sigma`, nie bierze wartości bezwzględnej z DOS i nie interpoluje brakujących
punktów. Punkty niezbieżne i nieprzyczynowe dostają jawny status i nie są
dołączane do scalonego `summary.csv`.

## Konwencje fizyczne

Domyślnie `D=0.5`, `t*=0.25`, `D=2 t*`, `w1=0.5`. Parametr
`disorder_full_width` jest pełną szerokością jednorodnego rozkładu
`[-Delta/2, Delta/2]`. Jednostki to `k_B=e=1`; ogólny prefaktor transportowy
jest wyłączony, więc `sigma` i `kappa_e` są w jednostkach naturalnych.

Funkcja transportowa jest liczona bezpośrednio dla ogólnego `D`:

```text
tau(omega) = (1/3) integral d epsilon rho0(epsilon)
             (D^2-epsilon^2) A(epsilon,omega)^2
```

Przy half-fillingu pojedyncze rozwiązanie spektralne `(U, Delta, branch)` jest
używane do wszystkich temperatur. Poza half-fillingiem każda ewaluacja
bracketowanej bisekcji po `mu` wykonuje pełny DMFT/TMT, korzystając z warm-startu
i cache.

## Instalacja i testy

W katalogu projektu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Jedyną obowiązkową zależnością jest NumPy. YAML, wykresy i pytest są opcjonalne:

```bash
python -m pip install -e '.[yaml,plot,test]'
```

Walidacja numeryczna:

```bash
python -m fk_transport validate --config configs/validation.json
```

Zapisuje `results/validation/validation_report.json`. Przed dużym skanem należy
osobno sprawdzić zbieżność względem `n_omega`, `omega_max`, `broadening`, obu
kwadratur, mieszania, tolerancji i `rho_floor`.

## Interfejs

```bash
python -m fk_transport solve --config configs/validation.json --U 0.3 --disorder 1.8 --branch arith
python -m fk_transport manifest --config configs/pilot_half_filling.json
python -m fk_transport sweep --config configs/pilot_half_filling.json --index 0
python -m fk_transport status --config configs/pilot_half_filling.json
python -m fk_transport merge --config configs/pilot_half_filling.json
python -m fk_transport plot --config configs/pilot_half_filling.json
```

`manifest` podaje liczbę zadań i ostatni indeks tablicy. `status` klasyfikuje
punkty jako `success`, `not_converged`, `noncausal`, `corrupt`, `missing` albo
`config_mismatch` i zapisuje `rerun_indices.txt`.

## Kruk / PBS

Na klastrze skopiuj cały katalog `fk_transport`, utwórz środowisko Pythona i
zainstaluj pakiet. Z katalogu projektu:

```bash
module load python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
PYTHON_EXECUTABLE="$PWD/.venv/bin/python" bash jobs/submit_pbs_array.sh configs/pilot_half_filling.json
```

Skrypt najpierw tworzy deterministyczny manifest, a następnie wysyła tablicę
PBS `0-(N-1)`. Jeden indeks odpowiada dokładnie jednemu punktowi spektralnemu
`(U, disorder_full_width, branch)` przy half-fillingu. Każdy punkt ma osobny
katalog, atomowy zapis i blokadę chroniącą przed dwoma jednoczesnymi writerami.

Po zakończeniu tablicy:

```bash
CONFIG=configs/pilot_half_filling.json \
PYTHON_EXECUTABLE="$PWD/.venv/bin/python" qsub jobs/merge_after_pbs.sh
```

Po skonfigurowaniu na Kruku klucza SSH z prawem zapisu do repozytorium można
zamiast zwykłego merge wysłać zadanie publikujące:

```bash
CONFIG=configs/pilot_half_filling.json \
PYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
RUN_LABEL=pilot_half_filling qsub jobs/publish_results_to_github.sh
```

Wyniki trafiają do osobnej gałęzi `results`, więc lokalna gałąź `main` zawiera
wyłącznie programy. Domyślnie publikowane są scalony `summary.csv`, status,
manifest, konfiguracja i raport walidacji. Pełny katalog wynikowy można wysłać
przez `PUBLISH_MODE=full`, o ile żaden plik nie przekracza 95 MB. Duże skany
powinny korzystać z magazynu danych lub Git LFS, ponieważ zwykłe repozytorium
GitHub nie jest przeznaczone do wielkich tablic numerycznych.

Jeżeli instalacja Kruka używa starego Torque zamiast PBS Pro i nie rozpoznaje
`qsub -J`, zamień w `jobs/submit_pbs_array.sh` opcję `-J` na `-t`; worker
obsługuje zarówno `PBS_ARRAY_INDEX`, jak i `PBS_ARRAYID`.

## Dane wyjściowe

Każdy `points/point_NNNNNN/` zawiera:

- `solution.npz` — siatkę oraz zespolone `hybridization`, `G`, `Sigma`, DOS i `tau`;
- `spectral.csv.gz` i `transport.csv.gz` — kolumnowe krzywe do dalszej analizy;
- `observables.json` — `L11`, `L12`, `L22`, `sigma`, `S`, `kappa_e`, liczbę Lorenza i flagi;
- `convergence.csv` — historię residuum, sum rule, minimum DOS, próg TMT i przyczynowość;
- `config.resolved.json` i `metadata.json` — pełną konfigurację, hash, wersje i status.

`metadata.json` jest zapisywany na końcu i stanowi znacznik ukończenia. Merge
weryfikuje hash konfiguracji. Duża liczba Lorenza przy `L11` poniżej progu jest
oznaczana jako `ill_conditioned`, a nie prezentowana jako wynik fizyczny.

## Konfiguracje

- `configs/validation.json` — szybkie testy czystego limitu;
- `configs/pilot_half_filling.json` — pilot z notatki, w tym `U=0.3`, `Delta=1.8`;
- `configs/production_half_filling.json` — gęstsza siatka produkcyjna;
- `configs/pilot_doped.json` — `n_c=0.45, 0.40, 0.35, 0.30` z doborem `mu`.

Konfiguracja produkcyjna jest punktem wyjścia, nie uniwersalnym certyfikatem
zbieżności. Tolerancję przyczynowości należy ustalić na podstawie kontroli
dyskretyzacji; surowe maksimum `Im Sigma` jest zawsze zapisane w metadanych.
