# pia_symbol_recognition

OPRAVIT CHYBY DO PREZENTACE - Neuronová síť nedává správné výsledky - do finální prezentace najít a opravit proč
                            - Vytvořit hlavičkové soubory
                            - vyrobit pořádnou dokumentaci
                            - gui zkontroluje zda je vytvořena síť

POSTUP POUŽITÍ
1) Skript main vytváří neuronovou síť ze zadaných parametrů a natrénuje ji pomocí dat MNIST (ta je momentálně chybná, ale vyustí v data ve správném formátu - parametry sítě uloží do souboru network.bin ) 
2) Skript callthis je volán pythonovským gui. -> načte síť z network.bin a na základě vektoru z obrázku poslaným z gui vypočte vektor výsledků, který pošle zpět k zobrazení do gui
3) Pokud je vytvořena síť (existuje network.bin), uživateli stačí spustit a používat pouze gui (python gui.py)

base -> obsahuje random pomocné funkce
loading -> metody a třídy pro načtení trénovacích dat
training -> metody pro trénink sítě a třída sítě
callthis -> vyhodnocovací skript pro specifický příklad zadaný uživatelem (otevírá ho gui)
main -> použít je jednou pro tvorbu sítě
