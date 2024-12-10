# pia_symbol_recognition

POZNÁMKA - NEURONOVÁ SÍŤ NEDÁVÁ SPRÁVNÉ VÝSLEDKY, NEMŮŽEME NAJÍT CHYBU

Popis souborů
1) Skript main vytváří neuronovou síť ze zadaných parametrů a natrénuje ji pomocí dat MNIST (ta je momentálně chybná, ale vyustí v data ve správném formátu - parametry sítě uloží do souboru network.bin a network.text vhodným pro uživatelské přečtení ) 
2) Skript callthis je volán pythonovským gui. -> načte síť z network.bin a na základě vektoru z obrázku poslaným z gui vypočte vektor výsledků, který pošle zpět k zobrazení do gui
3) Pokud je vytvořena síť (existuje network.bin), uživateli stačí spustit a používat pouze gui (python gui.py)

base -> obsahuje random pomocné funkce
loading -> metody a třídy pro načtení trénovacích dat
training -> metody pro trénink sítě a třída sítě
callthis -> vyhodnocovací skript pro specifický příklad zadaný uživatelem (otevírá ho gui)
main -> použít je jednou pro tvorbu sítě

NÁVOD K POUŽITÍ:
1) Zkompilovat a spustit soubor main.cpp -> vytvoří neuronovou síť 
2) Spustis soubor gui.py -> vyhodnotí nakreslené číslice dle neuronové sítě
