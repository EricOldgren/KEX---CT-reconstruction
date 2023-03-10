## Analytisk rekonstruktion för 2D tomografi med djupinlärning

Bakgrund: Tomografi är samlingsnamnet på en rad teknologier för att visualisera den interna strukturen av ett objekt. Metoderna bygger på att i en tomograf upprepade gånger genomlysa objektet ifråga med genomträngande strålning/våg från olika riktningar. Resulterande mätdata kan nu efter en del modellering tolkas som brusiga indirekta observationer av objektets interna struktur. Speciellt utvecklade matematiska metoder (tomografisk rekonstruktion) används därefter för att matematiskt återskapa en bild av objektets interna struktur utifrån sådana mätdata.

Tomografi är idag en oumbärlig teknologi inom medicin, biologi, arkeologi, geofysik, oceanografi, materialvetenskap, mätteknik (metrologi), astrofysik och andra vetenskaper. Det kanske mest kända exemplet är datortomografi inom medicinsk bilddiagnostik som belönades redan 1979 med Nobelpriset i fysiologi eller medicin. Genomlysningen bygger på samma princip som en konventionell röntgen, men istället för att bara ta en bild, kommer röntgenkameran i datortomografi att rotera runt och längsmed patienten som en spiral och ta en mängd röntgenbilder (=data) från olika lägen. Speciellt anpassade matematiska metoder kan nu utifrån dessa röntgenbilder skapa en 3D bild av patientens inre (= objektets interna struktur). Denna 3D bild av anatomin, vilket skiljer sig från konventionell röntgenundersökning som ger en 2D bild där den anatomisk informationen är överlagrad. Information man får från 3D bilden är till stor nytta då man exempelvis vill studera lokalisering av tumörer, blödningar, benfraktioner, dislokationer och ligamentskador. Datortomografi bilder ger också en hög kontrastupplösning, vilket gör att man lättare kan skilja mellan olika typer av vävnader vilket är viktigt då man vill upptäcka tumörer och blödningar.

Vetenskaplig frågeställning: De analytiska metoderna som används inom datortomografi är speciellt anpassade till hur data samlas in. I synnerhet utgår de ifrån att röntgenkameran kan rotera runt hela patienten. Det finns en del situationer där detta inte är möjligt, exempelvis då röntgenkameran enbart får rotera inom ett begränsat vinkelområde. Det finns idag inga kända analytiska metoder som är anpassade för sådana data. 

Den vetenskapliga frågeställningen avser att undersöka om djupinlärning kan kombineras med analytisk rekonstruktion för data med begränsat vinkelområde. Sådana metoder skulle öppna upp för helt nya möjligheter till att bedriva röntgendiagnostik. Utöver tillämpningar inom medicinsk bilddiagnostik kan de matematiska metoderna även nyttjas inom andra områden där den interna strukturen av ett objekt skall bestämmas från tomografiskt data med begränsat vinkelområde. Exempel på sådana områden är inom radar och oljeprospektering samt mikroskopi.

Projekt: Detta projekt kommer bedrivas i 2D fallet. Målet är att först formulera det statistiska inlärningsproblemet som svarar mot en analytiska rekonstruktionsmetod med inlärda filter. Därefter skall denna implementeras i PyTorch. 

Förkunskapskrav: Projektet är både matematiskt och implementationsmässigt krävande. För att minimera programmeringen kommer implementationen att baseras på ett existerande ramverk för att representera och hantera tomografisk data som även har bindningar till PyTorch. Att förstå logiken i detta ramverk och koppla dess datastrukturer till både PyTorch och de matematiska operatorerna kräver dock god förtrogenhet med programmering. Det går däremot att genomföra projektet utan att förstå den matematiska teorin (integralgeometri) som ligger till grund för de analytiska rekonstruktionsmetoderna. Dock behövs viss förtrogenhet med djupinlärning och goda förkunskaper inom flervariabelanalys.


## Preliminärt schema:

* Januari 17: kl 08-10. Uppstartsmöte del 1 för studenter: gemensamt för
SCI-skolan (handledare behöver ej närvara)
* Januari 17: kl 10-11. Uppstartsmöte del 2: KEX @ NA handledare och
studenter
* Februari 1: Titel & innehåll/projektplan i punktform (högst en sida,
helst PDF-fil) till handledare och eliasj@kth.se
* April 1: Studenter bör ha börjat med rapporten
* April 29: Första preliminär version av rapport till handledare (+allmän
återkoppling)
* Maj 5: Kamratgranskning: Preliminär rapport skickas för kamratgranskning
* Maj 8, 08-13 och Maj 9, 08-13.  Presentation av NA KEX för  KEX @ NA via
zoom. Gemensamt med KEX på optsyst.
* Maj 12: Kamratgranskning: Återkoppling till student
* Maj 16-17: Presentationsdagar på SCI-skolan
* Maj 19: Korrigerad rapport lämnas till handledare
* Maj 26: Återkoppling från handledare
* Juni 7: Inlämning av slutrapporter samt självvärderingsprotokoll till
kansliet (hård gräns)
