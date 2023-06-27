## Analytisk rekonstruktion för 2D tomografi med djupinlärning

## Sommarprojekt

* [Datamängd 1: Limited angle tävling](https://zenodo.org/record/6937616) och [Datamängd 2: Generell CT data](https://arxiv.org/pdf/2306.05907.pdf)
    * Data 2 innehåller mer realistiska exempel, medan Data 1 är binär (1 eller 0).
    * Gissningsvis kommer det bli svårt att matcha en Radontransform till mätningarna, så det lär vara bättre att bara använda fantomerna och generera mätdata själva (dvs släng Y och generera själva Y från X som AX + brus).
* [Constrained Empirical Risk Minimisation](https://arxiv.org/abs/2302.04729)
* [ Snabb Analytisk Extrapolering via Range Conditions](https://iopscience.iop.org/article/10.1088/2057-1976/aa71bf)
* Natterer tar också upp en snarlik metod i kapitel 6 i [en av böckerna som Ozan tipsade om](https://epubs.siam.org/doi/book/10.1137/1.9780898719284)
  * Range conditions som projektion på polynom
  * Utvidga momentprojektionerna trigonometriskt
  * Momentkurvorna ger sinogrammet 
