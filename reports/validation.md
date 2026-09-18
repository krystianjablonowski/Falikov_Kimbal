# Raport walidacyjny

Ten plik jest dziennikiem fazy walidacyjnej. Wartości liczbowe generuje polecenie:

```bash
python -m fk_transport validate --config configs/validation.json
```

Raport maszynowy jest zapisywany jako `results/validation/validation_report.json`.

## Zakres

- nieparzysta, symetryczna siatka z dokładnym `omega=0`;
- normalizacja półokrągłej DOS dla ogólnego `D`;
- czysty limit arytmetycznej DMFT;
- zgodność gałęzi arytmetycznej i typowej przy zerowym nieporządku;
- niezależne transformaty Hilberta FFT i kwadraturowa;
- parzystość transportu i zanik `L12` przy half-fillingu;
- dodatniość funkcji transportowej i nierówność Cauchy'ego--Schwarza.

## Warunek przejścia do produkcji

Duży skan należy uruchomić dopiero po przejściu walidacji oraz osobnym badaniu
zbieżności względem `n_omega`, `omega_max`, `broadening`, obu kwadratur,
`mixing`, tolerancji DMFT/TMT i `rho_floor`. Punkty `not_converged` i
`noncausal` nie trafiają do `summary.csv`.

## Wynik lokalnego uruchomienia referencyjnego

Środowisko: Python 3.12.14, NumPy 2.3.5. Konfiguracja walidacyjna:
`D=0.5`, `t*=0.25`, `n_omega=1001`, `omega_max=2.5`,
`broadening=0.005`, 32 węzły nieporządku i 96 węzłów pasmowych.

- 11/11 testów jednostkowych przeszło;
- walidacja CLI: wszystkie 11 kontroli przeszło;
- czysty limit arytmetyczny: względny błąd `G` `4.04e-16`;
- całka DOS Bethego: `1.0000000039`;
- różnica DOS gałęzi przy `Delta=0`: `9.63e-5`;
- oba punkty walidacyjne zapisane, status: 2 `success`, 0 braków.

Dodatkowy punkt diagnostyczny z notatki, `U=0.3`, pełna szerokość
`Delta=1.8`, `T=0.01`, `mu=U/2`:

| gałąź | L11 | L22 | kappa_e | Lorenz | status |
|---|---:|---:|---:|---:|---|
| arith | 1.85796e-2 | 6.10991e-6 | 6.10991e-4 | 3.28850 | success |
| typ | 4.44018e-4 | 1.43037e-7 | 1.43037e-5 | 3.22143 | success |

Jest to smoke test na zgrubnej siatce i z regulatorem większym niż w produkcji,
nie końcowy benchmark. Wynik arytmetyczny jest bliski wartościom starego kodu,
ale zgodność nie została wymuszona żadnym obcięciem ani kalibracją.
