# Analytisk rekonstruktion för 2D tomografi med djupinlärning

## Sommarprojekt

* [Datamängd 1: Limited angle tävling](https://zenodo.org/record/6937616) och [Datamängd 2: Generell CT data](https://arxiv.org/pdf/2306.05907.pdf)
    * Data 2 innehåller mer realistiska exempel, medan Data 1 är binär (1 eller 0).
    * Gissningsvis kommer det bli svårt att matcha en Radontransform till mätningarna, så det lär vara bättre att bara använda fantomerna och generera mätdata själva (dvs släng Y och generera själva Y från X som AX + brus).
 
### Litteratur
* [Constrained Empirical Risk Minimisation](https://arxiv.org/abs/2302.04729)
* [Enforcing constraints for interpolation and extrapolation in Generative Adversarial Networks](https://www.sciencedirect.com/science/article/pii/S0021999119305285) - kanske relevant
* [ Snabb Analytisk Extrapolering via Range Conditions](https://iopscience.iop.org/article/10.1088/2057-1976/aa71bf) - Natterer tar också upp en snarlik metod i kapitel 6 i [en av böckerna som Ozan tipsade om](https://epubs.siam.org/doi/book/10.1137/1.9780898719284). Metoden i korthet:
   * Omformulare Range conditions som projektion på system av ortogonala polynom
   * Utvidga momentprojektionerna trigonometriskt baserat på att endast ett ändligt antal fourier koefficienter är nollskilda
   * Beräkna sinogrammet från de utvidgade momentkurvorna
 
### Sinogramextrapolering - Fan Beam

#### Momentvillkor formulering för ortogonala polynom
Momentvillkoren lyder: funktionen $g(\varphi, s)$ ligger i bildrummet till Radontransformen omm

$$
   a_n(\varphi) = \int_{-\infty}^\infty g(\varphi, s)s^n ds
$$

är ett homogent polynom i $\sin$ och $\cos$ av grad $n$ för alla naturliga tal $n$. Dvs för varje $n$ är

$$
   a_n(\varphi) = \sum_{k=0}^n b_k \sin^k(\varphi)\cos^{n-k}(\varphi)
$$

exempelvis för $n = 2$ betyder detta att $a_2$ är en linjär kombination av funktionerna $\sin^2(\varphi)$, $\sin(\varphi)\cos(\varphi)$ och $\cos^2(\varphi)$. Funktionerna kan skrivas om

$$
   \cos^k(\varphi) = (\frac{e^{i\varphi} + e^{-i\varphi}}{2})^k =
   \frac{1}{2^k} \sum_{l=0}^k
   \left(\begin{array} .k \\
   l \end{array}\right\) e^{-il\varphi}e^{i(k-l)\varphi} =
   \frac{1}{2^k} \sum_{l=0}^k
   \left(\begin{array} .k \\
   l \end{array}\right\) e^{i(k-2l)\varphi}
$$

et.c. Så man kan visa att en ekvivalent bas för rummet av homogena polynom av grad $n$ i $\sin$ och $\cos$ är { $e^{ik\varphi}: |k|\leq n, k + n \text{ jämnt}$ }. En formulering av momentvillkoren är alltså

$$
   a_n(\varphi) = \sum_{|k|\leq n \land k + n \equiv 0 \pmod{2}} c_k e^{ik\varphi}
$$

där alltså $k = -n, -n+2,..., n-2, n$. Alternativt formulerat: Fourier koefficienterna $c_k$ till $a_n$ uppfyller $c_k = 0$ för $|k|>n$ och $k+n$ udda. I den här formuleringen framgår att om varje polynom $s^n$ byts ut mot valfritt polynom av grad $n$ sådant att polynomet är udda om $n$ är udda och jämnt om $n$ är jämnt blir momentvillkoren desamma. Därför kan vi byta ut $s^n$ mot polynom som är ortogonala på $[-1,1]$, exempelvis Chebyshev polynom.

#### Serieutveckling av bildrummet till $R$
Vi väljer en familj, $U_n(s)$ av polynom som momentvillkoren är giltiga för sådana att de är ortogonala under någon viktfunktion $W(s)$ på intervalllet $[-1,1]$

$$
   \int_{-1}^{1} U_n(s)U_m(s)W(s) ds = \delta_{m. n} \cdot ||U_m||
$$

exempelvis Chebyshev med $W(s) = \sqrt{1 - s^2}$.
Vi antar att våra objekt har stöd inuti enhetsdisken $||x|| < 1$ så att sinogrammen bara är nollskillda på området $0\leq \varphi < 2\pi$, $|s| < 1$. Eftersom polynom är täta i $L^2 [-1,1]_{W(s)}$ och spannet av $e^{ik\varphi}$ är tätt i $L^2[0,2\pi]$ kan en godtycklig funktion $g \in L^2[0,2\pi]\times[-1,1]$ serie utvecklas enligt (visst stämmer detta??!)

$$
   g(\varphi, s) / W(s) = \sum_n \sum_k c_{n, k} U_n(s) e^{ik\varphi} \iff \\
   g(\varphi, s) = \sum_n \sum_k c_{n, k} U_n(s) e^{ik\varphi}W(s)
$$

Basfunktionerna $U_n(s)e^{ik\varphi}$ är ortogonala enligt

$$
   \int_0^{2\pi}\int_{-1}^1 U_m(s)e^{ik\varphi}U_n(s)e^{-il\varphi}W(s) dsd\varphi = \left(\int_{-1}^1U_m(s)U_n(s)W(s)ds \right)\left( \int_0^{2\pi} e^{i(k-l)\varphi} d\varphi \right) = \delta_{m,n}\delta_{k,l} 2\pi ||U_m||
$$

så koefficienterna $c_{n,k}$ ges av projektionerna

$$
   c_{n, k} = \int_0^{2\pi}\int_{-1}^1 \left( g(\varphi, s) / W(s) \right) U_n(s) e^{-ik\varphi} ds d\varphi = \int_0^{2\pi} e^{-ik\varphi} \left( \int_{-1}^1 g(\varphi, s) U_n(s) ds\right) d\varphi = \int_0^{2\pi} a_n(s) e^{-ik\varphi} d\varphi
$$

dvs den $k$:te Fourier koefficienten till $a_n$, så $g$ uppfyller momentvillkoren om och endast om $c_{n, k}$ är nollskillda endast då $|k|\leq n$ och $k+n$ är jämnt.

Det betyder att vi kan använda dessa basfunktioner för att generera sinogram, projicera ner på det giltiga delrummet av sinogram et.c.


