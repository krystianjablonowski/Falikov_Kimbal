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

- `transport_heatmap_sigma_T0.pdf` — `sigma(T -> 0) = tau(omega=0)` dla DMFT i TMT;
- `transport_heatmaps_T_0p01.pdf` i `transport_heatmaps_T_0p02.pdf` — skończone
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
  --thresholds 1e-2 1e-4 1e-6 \
  --bandwidth 1.0 \
  --max-boundary-separation 0.1 \
  --ambiguity-tolerance 0.025
```

Komenda zapisuje `combined_summary.csv`, `relative_boundary_crossings.csv`,
`relative_boundary_differences.csv`, `gradient_boundaries.csv` oraz wykresy
granic wyznaczonych z bezwymiarowych ilorazów `sigma_typ/sigma_arith` i
`kappa_typ/kappa_arith`. Kilka progów jest celowe:
stabilność linii względem progu pozwala odróżnić fizyczną granicę od arbitralnej
definicji numerycznego zera.
Plik `relative_boundary_level_status.csv` podaje dla każdego progu zakres
wartości ilorazu, informację czy kontur istnieje oraz liczbę jego spójnych
składowych. Próg leżący poniżej minimum danych jest jawnie oznaczony jako
nieosiągnięty, a nie sztucznie dorysowywany.
Różnica położenia granic jest liczona tylko dla jednoznacznych par przecięć
oddalonych o nie więcej niż `max-boundary-separation`. Odległe lub konkurencyjne
dopasowania są pomijane. Statystyki zaakceptowanych, odrzuconych i
niejednoznacznych par oraz medianę i 90. percentyl separacji zapisuje
`relative_boundary_separation_summary.csv`.

### Etap 3: test granicy `eta -> 0`

Po ponownym wykonaniu `analyze-boundaries` generator wybiera punkty z dużą i
małą separacją, a następnie dobiera istniejące punkty siatki po obu stronach
konturu:

```bash
python -m fk_transport prepare-convergence \
  --config configs/stage1_half_filling_refined.json \
  --combined-summary results/stage2_boundaries/combined_summary.csv \
  --boundary-differences results/stage2_boundaries/relative_boundary_differences.csv \
  --output-prefix configs/stage3_convergence \
  --maximum-points 36 --neighbors 3 \
  --temperatures 0.005 0.01 0.03 \
  --threshold 1e-2
```

Powstają trzy konfiguracje dla `(eta,n_omega)` równego `(5e-4,20001)`,
`(2.5e-4,40001)` i `(1.25e-4,80001)`. Każda liczy te same punkty, więc zmianę
wyniku można przypisać kontrolowanemu limitowi poszerzenia i rozdzielczości.
Ze względu na koszt siatki `80001` należy użyć jednego punktu na job:

```bash
PYTHON_EXECUTABLE="$PWD/.venv/bin/python" bash jobs/submit_stage3_convergence.sh
```

Po ukończeniu należy dla każdej konfiguracji wykonać `status` i `merge`, a
następnie porównać trzy podsumowania:

```bash
python -m fk_transport analyze-convergence \
  --summaries \
    results/stage3_convergence/eta_5e-4/summary.csv \
    results/stage3_convergence/eta_2p5e-4/summary.csv \
    results/stage3_convergence/eta_1p25e-4/summary.csv \
  --etas 5e-4 2.5e-4 1.25e-4 \
  --output-directory results/stage3_convergence/analysis \
  --threshold 1e-2 --max-boundary-separation 0.1
```

Analizę publikuje niezależny skrypt:

```bash
SOURCE_DIRECTORY=results/stage3_convergence/analysis \
RUN_LABEL=stage3_convergence_analysis RESULTS_BRANCH=results \
bash jobs/publish_analysis_to_github.sh
```

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
Zadania proszą domyślnie o 12 godzin walltime i używają oszczędnego trybu
`FK_OUTPUT_MODE=compact`.

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

Domyślny tryb `compact` zapisuje w każdym `points/point_NNNNNN/`:

- `solution.npz` — siatkę oraz zespolone `hybridization`, `G`, `Sigma`, DOS i `tau`;
- `observables.json` — `L11`, `L12`, `L22`, `sigma`, `S`, `kappa_e`, liczbę Lorenza i flagi;
- `metadata.json` — hash konfiguracji, wersje i status.

To wystarcza do `status`, `merge` i późniejszych analiz widmowych. Tryb
`FK_OUTPUT_MODE=full` dodaje redundantne `spectral.csv.gz`, `transport.csv.gz`,
`convergence.csv` i kopię konfiguracji. Już wykonane punkty można bezpiecznie
odchudzić, zachowując `solution.npz`, poleceniem:

```bash
python -m fk_transport compact-results --config configs/pilot_half_filling.json
```

Generatory wykresów zapisują wyłącznie pliki PDF.

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
- `configs/stage4_filling_0p4.json` — regularny pilot `9 x 12 x 2 x 4`
  dla `n_c=0.4`; 864 zadania spektralne, grupowane po trzy w 288 jobów PBS.
- `configs/stage4_filling_0p4_expanded.json` — rozszerzona siatka `17 x 21`
  dla `n_c=0.4`, z `U/W` i `Delta/W` do `3`; 2856 zadań spektralnych,
  grupowanych po osiem w 357 jobów PBS.
- `configs/stage5_filling_0p3_dense.json` — gęsta siatka `25 x 25` dla
  `n_c=0.3`, z `U/W` i `Delta/W` od `0` do `3` co `0.125`; 5000 zadań
  spektralnych, grupowanych po dziesięć w 500 jobów PBS.
- `configs/stage6_filling_0p2_dense.json` — taka sama gęsta siatka dla
  `n_c=0.2`, z rozszerzonym przedziałem bisekcji potencjału chemicznego;
  5000 zadań spektralnych, grupowanych po dziesięć w 500 jobów PBS.
- `configs/stage7_filling_0p45_dense_multitemp.json` — siatka `25 x 25` dla
  `n_c=0.45`, z `U/W` i `Delta/W` od `0` do `3` co `0.125` oraz temperaturami
  `T/W=0.005, 0.01, 0.02, 0.05, 0.1, 0.5`; 7500 zadań spektralnych,
  grupowanych po dziesięć w 750 jobów PBS.
- `configs/stage8_filling_0p25_dense.json` — taka sama gęsta siatka dla
  `n_c=0.25` i temperatur `T/W=0.005, 0.01, 0.02, 0.05`; 5000 zadań
  spektralnych, grupowanych po dziesięć w 500 jobów PBS.
- `configs/stage9_fillings_near_half_T_0p05.json` — sześć nowych wypełnień
  `0.49, 0.475, 0.425, 0.375, 0.35, 0.325` na siatce `37 x 37`, przy jednej
  temperaturze `T/W=0.05`; istniejące `n_c=0.45` pochodzi ze skanu stage 7.
  Nowy etap ma 16428 zadań spektralnych, grupowanych po 24 w 685 jobów PBS.

Analiza niepołowicznego wypełnienia z istniejącego `summary.csv`:

```bash
python -m fk_transport analyze-filling \
  --input results/stage4_filling_0p4_expanded/summary.csv \
  --output-directory results/stage4_filling_0p4_expanded \
  --bandwidth 1.0
```

Testy numeryczne głównej tezy publikacyjnej można wykonać bez ponownego DMFT,
korzystając z `summary.csv` i zachowanych `solution.npz`:

```bash
python -m fk_transport publication-tests \
  --summaries results/stage5_filling_0p3_dense/summary.csv \
  --points-roots results/stage5_filling_0p3_dense/points \
  --output-directory results/publication_tests \
  --temperatures 0.01 0.02 \
  --localization-threshold 1e-4 \
  --conductivity-relative-floor 1e-4 \
  --u-min 1.5
```

Polecenie porównuje linie `S=0` z granicą lokalizacji, dopasowuje
`Delta_c=Delta_infinity+C/U^2`, sprawdza dokładną relację kowariancyjną dla
`S_typ-S_arith`, porównuje termosiłę Kubo i Kelvina oraz eksportuje liczbę
Lorenza. Można podać wiele plików w `--summaries` i tyle samo katalogów w
`--points-roots`, aby wspólnie przeanalizować różne wypełnienia.

Punkty, dla których przewodność spada poniżej podanej części maksymalnej
przewodności danej gałęzi, są zachowane w CSV, ale pomijane na wykresach
termosiły jako numerycznie niestabilne. Plik
`localization_boundary_coverage.csv` wyjaśnia dla każdego `U`, czy zadany próg
lokalizacji został przecięty, nie został osiągnięty, czy cały dostępny zakres
leży już poniżej progu. Dzięki temu brak krzywej nie daje pustego wykresu bez
diagnozy. Wykres kowariancji jest testem tożsamości algebraicznej i służy do
walidacji implementacji, a nie jako niezależny wynik fizyczny.

Polecenie zapisuje względne zanikanie `typ/arith`, moc termoelektryczną,
elektronowe `ZT` oraz wartości własne sprzężonej macierzy dyfuzji `D_-`, `D_+`.

Linie kompensacji termoelektrycznej `L12=0` można przeanalizować bez ponownego
rozwiązywania DMFT:

```bash
python -m fk_transport analyze-compensation \
  --input results/stage5_filling_0p3_dense/summary.csv \
  --points-root results/stage5_filling_0p3_dense/points \
  --output-directory results/stage5_filling_0p3_dense/compensation_analysis \
  --bandwidth 1.0
```

Bez `--points-root` powstają kontury dla wszystkich temperatur, ich porównanie
`arith`--`typ`, tabela sparowanych przecięć i miary odległości między liniami.
Opcjonalny katalog punktów dodaje rozkład `L12` na wkłady z `omega<0` i
`omega>0` oraz odległość L1 między znormalizowanymi rozkładami transportowymi
obu średnich. Wyniki są zapisywane jako CSV oraz rysunki PNG/PDF.

Profile funkcji transportowej w automatycznie wybranych punktach (strona
metaliczna, maksimum `|S_typ|`, linia kompensacji i krawędź lokalizacji):

```bash
python -m fk_transport plot-compensation-profiles \
  --input results/stage5_filling_0p3_dense/summary.csv \
  --points-root results/stage5_filling_0p3_dense/points \
  --output-directory results/stage5_filling_0p3_dense/compensation_profiles \
  --interactions 1.0 1.125 1.25 \
  --edge-ratio 1e-4 \
  --bandwidth 1.0
```

Każda plansza zestawia `tau(omega)`, znormalizowany rozkład transportowy,
całkę pod `L12` i jej skumulowaną kompensację dla średniej arytmetycznej i
typowej. Plik `compensation_profile_points.csv` dokumentuje automatyczny wybór
punktów, ich indeksy, `S`, `L12`, przewodności i stosunek `sigma_typ/sigma_arith`.

Konfiguracja produkcyjna jest punktem wyjścia, nie uniwersalnym certyfikatem
zbieżności. Tolerancję przyczynowości należy ustalić na podstawie kontroli
dyskretyzacji; surowe maksimum `Im Sigma` jest zawsze zapisane w metadanych.

### Kontrola numeryczna granicy lokalizacji

Oscylacje stosunku `rho_typ(0)/rho_arith(0)` wokół zadanego progu mogą
tworzyć wiele pozornych przecięć. Przed interpretacją takiej granicy należy
sprawdzić zbieżność względem poszerzenia `eta`, kroku siatki energii i rzędu
kwadratury nieporządku. Mały skan kontrolny dla `n_c=0.4`, `T/W=0.02`
przygotowuje polecenie:

```bash
python -m fk_transport prepare-localization-convergence \
  --config configs/stage4_filling_0p4_expanded.json \
  --diagnostics results/publication_tests_fillings_threshold_1e-2_fixed/publication_point_diagnostics.csv \
  --output-prefix configs/stage9_localization_convergence \
  --filling 0.4 \
  --temperature 0.02 \
  --interactions 0.75 1.5 2.25 3.0 \
  --ratio-target 1e-2 \
  --neighbors 3

BATCH_SIZE=4 PYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  bash jobs/submit_localization_convergence.sh
```

Powstają cztery konfiguracje: trzy testują malejące `eta` wraz z coraz
gęstszą siatką energii, a czwarta podwaja rząd kwadratury przy ustalonych
`eta` i siatce. Liczone są wyłącznie punkty typowe w sąsiedztwie granicy,
nie cała mapa.

Po zakończeniu jobów:

```bash
for cfg in configs/stage9_localization_convergence_*.json; do
  python -m fk_transport status --config "$cfg"
  python -m fk_transport merge --config "$cfg"
done

python -m fk_transport analyze-localization-convergence \
  --summaries \
    results/stage4_filling_0p4_expanded/summary.csv \
    results/stage9_localization_convergence/eta_5e-4_n20001_q96/summary.csv \
    results/stage9_localization_convergence/eta_2p5e-4_n40001_q96/summary.csv \
    results/stage9_localization_convergence/eta_1p25e-4_n80001_q96/summary.csv \
    results/stage9_localization_convergence/eta_5e-4_n20001_q192/summary.csv \
  --configs \
    configs/stage4_filling_0p4_expanded.json \
    configs/stage9_localization_convergence_eta_5e-4_n20001_q96.json \
    configs/stage9_localization_convergence_eta_2p5e-4_n40001_q96.json \
    configs/stage9_localization_convergence_eta_1p25e-4_n80001_q96.json \
    configs/stage9_localization_convergence_eta_5e-4_n20001_q192.json \
  --output-directory results/stage9_localization_convergence/analysis
```

Analiza zapisuje wyłącznie PDF oraz dwie tabele CSV: wszystkie wartości
punktowe i rozrzut między ustawieniami numerycznymi. Wyniki można wysłać na
gałąź `results` poleceniem:

```bash
SOURCE_DIRECTORY=results/stage9_localization_convergence/analysis \
RUN_LABEL=stage9_localization_convergence_analysis \
RESULTS_BRANCH=results \
bash jobs/publish_analysis_to_github.sh
```
