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

Polecenie `plot` tworzy osobne podsumowanie dla każdej temperatury i
wypełnienia. Dla każdej temperatury powstają również cztery zestawy map:

- `transport_heatmaps_*` — `sigma` i `kappa_e`;
- `thermodynamic_heatmaps_*` — elektronowe `c_V` i `K0_thermo`;
- `diffusivity_heatmaps_*` — diagnostyczne `D_c` i `D_E`;
- `diagnostic_heatmaps_*` — liczba Lorenza i wariancja energii transportowej.

Mapy używają wspólnej logarytmicznej normalizacji gałęzi arytmetycznej i TMT
oraz kontrastowej palety `inferno`. Opis osi `U/W` jest pokazywany tylko w lewej
kolumnie, a colorbary mają osobną kolumnę poza panelami danych.

### Etap 1: zależność temperaturowa i elektronowe ciepło właściwe

Każdy nowy punkt zapisuje dodatkowo transportową średnią i wariancję energii,
niezależną kontrolę tożsamości

```text
kappa_e = sigma * transport_energy_variance / T
```

oraz elektronowe ciepło właściwe przy stałej gęstości

```text
c_v_electronic = (K2_thermo - K1_thermo^2 / K0_thermo) / T.
```

Zapisywane są też ilorazy `charge_diffusivity_proxy = sigma/K0_thermo` oraz
`thermal_diffusivity_proxy = kappa_e/c_v_electronic`. Przy half-fillingu, gdy
`L12=0`, odpowiadają one rozprzężonym skalarnym dyfuzyjnościom. Poza
half-fillingiem są tylko diagnostyką; fizyczne mody dyfuzji wymagają pełnej
macierzy sprzężonego transportu ładunku i energii.

Termodynamika korzysta z arytmetycznej DOS, również dla kąpieli TMT. Formuła
dotyczy jednorodnej fazy z ustalonym `w1`; nie zawiera wkładu fononowego,
temperaturowej zmiany koncentracji cząstek nieruchomych ani uporządkowania CDW.

Konfiguracja `configs/stage1_half_filling.json` zawiera trzy reprezentatywne
wartości oddziaływania, przekrój przez nieporządek i gęstą listę temperatur.
Przy half-fillingu jedna konwergentna funkcja spektralna jest używana dla
wszystkich temperatur, więc zwiększenie liczby temperatur jest tanie.

Do map publikacyjnych służy `configs/stage1_half_filling_refined.json`: siatka
`25 x 31 x 2 = 1550` punktów spektralnych i ta sama lista ośmiu temperatur.
Przy `BATCH_SIZE=4` skan wymaga 388 zadań PBS.

Z już zapisanego punktu można obliczyć nową siatkę temperatur bez ponownego
uruchamiania DMFT/TMT:

```bash
python -m fk_transport reweight \
  --point results/stage1_half_filling/points/point_000000 \
  --temperatures 0.005 0.0075 0.01 0.015 0.02 0.03 0.05 0.08 \
  --plot
```

Powstają `temperature_scan.csv` oraz, z opcją `--plot`, wykres PNG/PDF. Dopasowanie
energii aktywacji dla przewodności elektrycznej i cieplnej wykonuje polecenie:

```bash
python -m fk_transport activation-fit \
  --input results/stage1_half_filling/points/point_000000/temperature_scan.csv \
  --fields sigma kappa_e \
  --temperature-min 0.005 --temperature-max 0.03
```

Plik `activation_fits.json` zawiera zwykły fit Arrheniusa oraz fit
`A*T^p*exp(-E/T)`. Ten drugi wymaga co najmniej trzech temperatur i powinien być
interpretowany razem z raportowanym uwarunkowaniem macierzy dopasowania.

Po scaleniu całego skanu można automatycznie dopasować każdy punkt
`(branch, U, Delta, filling)` bez mieszania różnych parametrów:

```bash
python -m fk_transport activation-map \
  --input results/stage1_half_filling/summary.csv \
  --fields sigma kappa_e c_v_electronic \
  --temperature-min 0.005 --temperature-max 0.03
```

Wyniki są zapisywane w `activation_summary.csv`.

`manifest` podaje liczbę zadań i ostatni indeks tablicy. `status` klasyfikuje
punkty jako `success`, `not_converged`, `noncausal`, `corrupt`, `missing` albo
`config_mismatch` i zapisuje `rerun_indices.txt`.

### Mapa transportowa odpowiadająca diagramowi Byczuka

Konfiguracja `configs/byczuk_transport_coarse.json` skanuje siatkę
`15 x 13 x 2 = 390` niezależnych zadań `(U, Delta, branch)` przy half-fillingu.
Siatka jest gęstsza w małych `U`, aby zawierała punkt `U/W=0.5`.
Po `merge` polecenie `plot` zapisuje nieinterpolowane mapy kolorów:

- `transport_heatmap_sigma_T0.png` — `sigma(T -> 0) = tau(omega=0)` dla DMFT i TMT;
- `transport_heatmaps_T_0p01.png` i `transport_heatmaps_T_0p02.png` — skończone
  temperaturowo `sigma` i `kappa_e`.

Osie są podane w jednostkach `W=2D`; w tej konfiguracji `W=1`. Cieplna
przewodność nie ma użytecznej mapy dokładnie w `T=0`, ponieważ wtedy znika
zarówno w metalu, jak i izolatorze. Brakujące albo odrzucone punkty pozostają
puste — program ich nie interpoluje. Jest to skan rozpoznawczy; przed publikacją
należy zagęścić okolice granic i wykonać test zbieżności parametrów numerycznych.

### Etap 2: adaptacyjne zagęszczenie granic

Polecenie `refine-boundaries` czyta regularny skan etapu 1 i tworzy nową
konfigurację zawierającą wyłącznie konkretne, nieregularnie rozmieszczone pary
`(U, Delta)`. Komórka coarse grid jest wybierana, jeśli zakres `log10` co
najmniej jednej obserwabli przekracza zadany `--log-jump`. Domyślnie granice są
wyznaczane z gałęzi `typ` w najniższej dostępnej temperaturze na podstawie
`sigma`, `kappa_e`, `charge_diffusivity_proxy` i
`thermal_diffusivity_proxy`. Jedna warstwa sąsiednich komórek jest dodawana jako
bufor.

Po pobraniu wyników etapu 1 na Kruku:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -m fk_transport refine-boundaries \
  --config configs/stage1_half_filling_refined.json \
  --input results/stage1_half_filling_refined/summary.csv \
  --output-config configs/stage2_boundaries.json \
  --output-directory results/stage2_boundaries \
  --u-step 0.025 --disorder-step 0.025 \
  --log-jump 0.75 --padding-cells 1 \
  --batch-size 8 --max-jobs 399
```

Kroki są podawane w jednostkach `U/W` i `Delta/W`. Program wypisuje liczbę
nowych par, punktów spektralnych, jobów PBS oraz minimalny `BATCH_SIZE`, który
utrzymuje tablicę pod limitem. Nie powtarza punktów obecnych w etapie 1.
Wygenerowany plik można przed wysłaniem sprawdzić:

```bash
python -m fk_transport manifest --config configs/stage2_boundaries.json
```

Jeśli raport zawiera `"within_job_limit": true`, obliczenia uruchamia się z tą
samą wielkością paczki, która została przekazana generatorowi:

```bash
BATCH_SIZE=8 PYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
bash jobs/submit_pbs_array.sh configs/stage2_boundaries.json
```

Jeśli `within_job_limit` jest fałszywe, należy użyć wartości wypisanej jako
`minimum_batch_size_for_limit`. Zwiększa to liczbę punktów liczonych kolejno w
jednym jobie, ale nie zmienia siatki ani wyników.

Po ukończeniu i scaleniu etapu 2 oba zestawy analizuje się wspólnie:

```bash
python -m fk_transport analyze-boundaries \
  --coarse results/stage1_half_filling_refined/summary.csv \
  --refined results/stage2_boundaries/summary.csv \
  --output-directory results/stage2_boundaries \
  --thresholds 1e-4 1e-6 1e-8 \
  --bandwidth 1.0
```

Komenda zapisuje `combined_summary.csv`, `boundary_crossings.csv`, adaptacyjne
mapy triangulowane oraz wykresy położeń wszystkich przecięć progowych dla
`sigma_typ` i `kappa_e_typ` w każdej temperaturze. Kilka progów jest celowe:
stabilność linii względem progu pozwala odróżnić fizyczną granicę od arbitralnej
definicji numerycznego zera.

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

Kilka punktów spektralnych można wykonać kolejno wewnątrz jednego zadania PBS.
Konfiguracja dokładna ma 1550 punktów, ale poniższe polecenie grupuje je po
cztery i wysyła tylko 388 zadań do kolejki:

```bash
BATCH_SIZE=4 PYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
bash jobs/submit_pbs_array.sh configs/byczuk_transport_refined.json
```

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
- `configs/byczuk_transport_coarse.json` — regularna mapa `(U/W, Delta/W)`;
- `configs/byczuk_transport_refined.json` — mapa `25 x 31 x 2`, grupowana w 388 jobów;
- `configs/pilot_filling_0p75.json` — mały test transportu dla `n_c=0.75`, `w1=0.5`;
- `configs/byczuk_transport_refined_filling_0p75.json` — mapa `25 x 31 x 2` dla `n_c=0.75`;
- `configs/pilot_doped.json` — `n_c=0.45, 0.40, 0.35, 0.30` z doborem `mu`.

Konfiguracja produkcyjna jest punktem wyjścia, nie uniwersalnym certyfikatem
zbieżności. Tolerancję przyczynowości należy ustalić na podstawie kontroli
dyskretyzacji; surowe maksimum `Im Sigma` jest zawsze zapisane w metadanych.
