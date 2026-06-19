# Opinia prowadzącego — dokumentacja końcowa

**Od:** Eryk
**Dotyczy:** dokumentacji końcowej projektu (`02_final_sn`)
**Ocena: 29/50**

---

Dzień dobry,

dziękuję za podesłanie dokumentacji końcowej projektu.

## Uwagi

- Opis problemu jest poprawny.

- Opis badanych metod jest ostatecznie bardzo lakoniczny: myślę, że tutaj lepiej
  sprawdziłby się pseudokod niż słowno-muzyczne opisanie badanych metod, bo np. na
  podstawie opisu metody naprawy punktów niedopuszczalnych nie da się stwierdzić,
  w jaki sposób dana permutacja jest przebudowywana.

- Nigdzie nie występuje definicja pojęcia "ścieżka maszynowa", której używacie
  np. na Rysunku 1.

- Nigdzie nie występuje rozwinięcie skrótu CPM, którego używacie, np. na Rysunku 1.

- Jest to chyba czwarta lub piąta praca, która używa korekcji Holma bez
  uzasadnienia, dlaczego w ogóle ta korekcja jest stosowana; uznaję to za LLMizm :)

- Co to jest ablacja hiperparametrów w kontekście algorytmów optymalizacyjnych?
  Chodzi Wam o tuning? Jeśli tak, to szkoda, że jej nie przeprowadziliście nawet
  na niewielkich grafach.

- Nigdzie nie są podane p-wartości, więc ciężko mi stwierdzić, że zdanie:

  > 25/25 istotnych testów na korzyść CPM-on, żaden przeciw (p_Holm ≤ 0,05).

  jest prawdziwe — szczególnie, że w większości przypadków mediana `pmx_cpm` jest
  niemal taka sama jak mediana pozostałych metod.

- Czym jest symbol `M` w tabeli nr 2?

- Prawdopodobny powód, dlaczego "marginesy bywają duże na trudnych grafach
  rzeczywistych", jest z sufitu: co Wam pozwala przypuszczać, że tak się w ogóle
  dzieje? Nic w tekście tego nie potwierdza.

- Nigdzie w tekście nie pojawia się informacja, jakim budżetem funkcji celu
  dysponowały konkretne solvery; jedynie w punkcie nr 2 w sekcji 5 ujawniacie, że
  budżet funkcji celu był różny dla badanych solverów, co stawia bardzo duży znak
  zapytania nad sensem raportowanych wyników: w zasadzie bez tej informacji można
  stwierdzić, że nie da się porównać badanych solverów…

## Podsumowanie

Ogólnie raport Waszej pracy wygląda z wierzchu bardzo profesjonalnie — gorzej
jest, gdy człowiek to próbuje czytać ze zrozumieniem, bo sałata słowna generowana
przez LLM-a jest bardzo ciężkostrawna. Na dodatek ciężko zauważyć przez to fakt,
że praca nie jest poprawna metodycznie.

**Ocena: 29/50**

Pozdrawiam,
Eryk
