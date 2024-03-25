"""
Fichiers d'exemples pour la problématique de l'APP6 (S2)
(c) Julien Rossignol & JB Michaud Université de Sherbrooke
v 1.1 Hiver 2024

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import helpers as hp


###############################################################################
def exampleRacines():
    """
    Calcule et affiche les pôles et zéros d'une FT arbitraire

    :return:
    """
    b1 = [1, 0, 0]  # définition des facteurs du polynome numérateur
    a1 = [1, 61224, 1943634596]  # définition des facteurs du polynome dénominateur

    # méthode plus manuelle
    z1 = np.roots(b1)
    # pour éventuellement faire l'inverse de l'opération roots, i.e. passer d'une
    # liste de racines aux coefficients du polynôme, voir np.poly
    p1 = np.roots(a1)
    print(f'Racine exemple 1 Zéros:{z1}, Pôles:{p1}')  # affichage du resultat dans la console texte
    # appel d'une fonction pour affichage graphique, voir autre fichier helpers.py
    hp.pzmap1(z1, p1, 'Pôles et zéro du passe haut de 7000Hz')

    b4 = [1, 0, 19340000]  # définition du numérateur de la fonction de transfert
    a4 = [1, 6220, 19340000]
    # méthode utilisant scipy
    (z2, p2, k2) = signal.tf2zpk(b4, a4)  # passage à la représentation zok (liste de racines + "gain")
    print(f'Racine exemple 2 Zéros : {z2}, Pôles: {p2}, Gain: {k2}')
    hp.pzmap1(z2, p2, 'Pôles et zéro du passe bas de 700Hz')
    b3 = [1, 0, 0]  # définition du numérateur de la fonction de transfert
    a3 = [1, 8878, 39400000]
    # méthode utilisant scipy
    (z3, p3, k3) = signal.tf2zpk(b3, a3)  # passage à la représentation zok (liste de racines + "gain")
    print(f'Racine exemple 2 Zéros : {z2}, Pôles: {p2}, Gain: {k2}')
    hp.pzmap1(z3, p3, 'Pôles et zéro du passe haut de 1000Hz')
    b5 = [1, 0, 986900000]  # définition du numérateur de la fonction de transfert
    a5 = [1, 44434, 986900000]
    # méthode utilisant scipy
    (z5, p5, k5) = signal.tf2zpk(b5, a5)  # passage à la représentation zok (liste de racines + "gain")
    print(f'Racine exemple 2 Zéros : {z5}, Pôles: {p5}, Gain: {k5}')
    hp.pzmap1(z5, p5, 'Pôles et zéro du passe bande de 5000Hz')


###############################################################################
def exampleBode():
    """
    Calcule et affiche le lieu de bode d'une FT arbitraire

    :return:
    """
    b1 = [1, 0, 0]  # définition du numérateur de la fonction de transfert
    a1 = [1, 61224, 1943634596]  # définition du dénominateur de la fonction de transfert

    # méthode 1 avec bode
    tf1 = signal.TransferFunction(b1, a1)  # définit la fonction de transfert
    # calcul le diagrame de Bode, la magnitude en dB et la phase en degrés la fréquence en rad/s
    w1, mag1, phlin1 = signal.bode(tf1, np.logspace(1, 5, 200))  # on pourrait aussi laisser la fonction générer les w
    # fonction d'affichage
    hp.bode1(w1, mag1, phlin1, 'Passe Haut 7000Hz')
    plt.show()
    b2 = [0, 0, 986900000]  # définition du numérateur de la fonction de transfert
    a2 = [1, 44434, 986900000]  # définition du dénominateur de la fonction de transfert
    tf2 = signal.TransferFunction(b2, a2)  # définit la fonction de transfert
    w2, mag2, phlin2 = signal.bode(tf2, np.logspace(1, 5, 200))
    hp.bode1(w2, mag2, phlin2, 'Passe Bas 5000Hz')
    plt.show()
    b3 = [1, 0, 0]  # définition du numérateur de la fonction de transfert
    a3 = [1, 8878, 39400000]  # définition du dénominateur de la fonction de transfert
    tf3 = signal.TransferFunction(b3, a3)  # définit la fonction de transfert
    w3, mag3, phlin3 = signal.bode(tf3, np.logspace(1, 5, 200))
    hp.bode1(w3, mag3, phlin3, 'Passe Haut 1000Hz')
    plt.show()
    b4 = [1, 0, 19340000]  # définition du numérateur de la fonction de transfert
    a4 = [1, 6220, 19340000]  # définition du dénominateur de la fonction de transfert
    tf4 = signal.TransferFunction(b4, a4)  # définit la fonction de transfert
    w4, mag4, phlin4 = signal.bode(tf4, np.logspace(1, 5, 200))
    hp.bode1(w4, mag4, phlin4, 'Passe Bas 700Hz')
    (z4, p4, k4) = signal.tf2zpk(b3, a3)
    #hp.bodeplot(tf4[0], tf4[1], "Passe Bas 700Hz")
    #mag, ph, w, fig, ax = hp.bodeplot(tf4[0], tf4[1], 'Hs(s)')

    plt.show()
###############################################################################
def FiltreApp():

    #Pour les K des potentiomètre, la valeur de K à été déterminer en faisant le calcul du gain d'un inverseur
    #qui est -R2/R1 ce qui implique que K1 = -47.5K/47.5K= -1
    #pour K2: -R2/R1 aussi ce qui implique que K2 = -47.5K/60.4K
    order = 2
    #Filtre passe bas 700Hz
    wc1 = 2*np.pi*700
    b1, a1 = signal.butter(order, wc1,analog=True,btype='lowpass' )
    z1, p1, k1 = signal.tf2zpk(b1, a1)
    #Filtre passe haut 1000Hz
    wc2 = 2 * np.pi * 1000
    b2, a2 = signal.butter(order, wc2, analog=True, btype='highpass')
    z2, p2, k2 = signal.tf2zpk(b2, a2)
    #Filtre passe bas 5000Hz
    wc3 =2*np.pi*5000
    b3, a3 = signal.butter(order, wc3, analog=True, btype='lowpass')
    z3, p3, k3 = signal.tf2zpk(b3, a3)

    #Filtre passe haut 7000Hz
    wc4 = 2 * np.pi * 7000
    b4, a4 = signal.butter(2, wc4, analog=True, btype='highpass')
    z4, p4, k4 = signal.tf2zpk(b4, a4)

    #mise en série du 1000Hz et du 5000Hz
    zs, ps, ks = hp.seriestf(z2, p2, k2, z3, p3,k3)

    #mise en parralèle du circuit 700 et 7000Hz
    z6, p6, k6 = hp.paratf(z1, p1, -1.00*k1, z4,p4,-1.00*k4)
    zf, pf, kf = hp.paratf(z6, p6, k6, zs, ps, 0.7864*ks)
    bf, af = signal.zpk2tf(zf, pf, kf)
    magf, phf, wf, figf, axf = hp.bodeplot(bf, af, "Lieu de bode à la sortie du circuit")
    hp.pzmap1(zf,pf,"Filtre audio")
    #délais de groupe


    # Reponse en frequence
    #magf, phf, w, fig, ax = hp.bodeplot(num1, denum1, 'egaliseur')
    #(zp, pz, kz) = signal.tf2zpk(b5, a5)  # passage à la représentation zok (liste de racines + "gain")
    #print(f'Racine exemple 2 Zéros : {z1}, Pôles: {p1}, Gain: {k1}')
    hp.pzmap1(zf, pf, 'Pôles et zéro du Circuit')

    #delay = - np.diff(ph1) / np.diff(w1)  # calcul
    # Delai de groupe
    delay = - np.diff(phf) / np.diff(wf)  # calcul
    hp.grpdel1(wf, delay, 'egaliseur')  # affichage
def exampleButterworth():
    """
    Exemple de génération et affichage pour la FT d'un filtre de butterworth d'ordre 4

    :return:
    """

    order = 4
    wn = 1   # frequence de coupure = 1 rad/s
    # définit un filtre passe bas butterworth =>  b1 numerateur, a1 dénominateur
    b1, a1 = signal.butter(order, wn, 'low', analog=True)
    print(f'Butterworth Numérateur {b1}, Dénominateur {a1}')  # affiche les coefficients correspondants au filtre
    print(f'Racine butterworth Zéros:{np.roots(b1)}, Pôles:{np.roots(a1)}')  # affichage du resultat dans la console texte


    # Réponse en fréquence
    mag1, ph1, w1, fig, ax = hp.bodeplot(b1, a1, 'Butterworth Example')

    # Délai de groupe
    delay = - np.diff(ph1) / np.diff(w1)    # calcul
    hp.grpdel1(w1, delay, 'Exemple butterworth')    # affichage

    # Exemple 1 réponse temporelle 3 cos
    t1 = np.linspace(0, 10, 5000, endpoint=False)
    # génération de 3 sinusoîdes
    u1 = (np.cos(2 * np.pi * 0.4 * t1) + 0.6 * np.cos(2 * np.pi * 4 * t1) +
          0.5 * np.cos(2 * np.pi * 8 * t1))
    # simulation temporelle du système
    tout1, yout1, xout1 = signal.lsim((b1, a1), u1, t1)
    hp.timeplt1(t1, u1, tout1, yout1, 'filtrage de 3 sinus')   # affichage

    # Exemple 2 réponse à l'échelon à la mitaine mitaine
    t2 = np.linspace(0, 10, 5000, endpoint=False)
    u2 = np.ones_like(t2)
    tout2, yout2, xout2 = signal.lsim((b1, a1), u2, t2)
    hp.timeplt1(t2, u2, tout2, yout2, 'échelon avec lsim')

    # Exemple 3 réponse à l'échelon via scipy.signal
    tout3, yout3 = signal.step((b1, a1))
    t3m = np.amax(tout3)
    stept3 = t3m/len(tout3)
    hp.timeplt1(np.arange(0, t3m, stept3), np.ones_like(tout3),
                tout3, yout3, 'échelon avec signal.step')


###############################################################################
def probleme1():
    """
    Framework pour le problème 1 du laboratoire

    :return:
    """
    #Question A
    
    ##FT1
    #Entrez les poles, les zeros et le gain de la FT #1
    z1 = [0]
    p1 = [-30612+31669j, -30612-31669j]
    k1 = 1
    
    #Ajouter le code pour obtenir la représentation de la fonction de transfert sous forme de polynome. Indice: Regarder la fonction scipy.signal.zpk2tf
    tf1 = signal.zpk2tf(z1, p1, k1)
    w1, mag1, phlin1 = signal.bode(tf1, np.logspace(0, 2, 200))

    #Ajouter le code pour generer la carte des poles et zeros et le lieu de Bode
    #utiliser les fonctions dans helpers.py
    # hp.pzmap1(z1, p1, "Lieu de bode H1(s)")
    # hp.bode1(w1, mag1, phlin1, "H1(s)")
    b1 = [1, 0, 0]  # définition du numérateur de la fonction de transfert
    a1 = [1, 1, 0.5]  # définition du dénominateur de la fonction de transfert

    # méthode 1 avec bode
    tf1 = signal.TransferFunction(b1, a1)  # définit la fonction de transfert
    # calcul le diagrame de Bode, la magnitude en dB et la phase en degrés la fréquence en rad/s
    w1, mag1, phlin1 = signal.bode(tf1, np.logspace(-1.5, 1, 200))  # on pourrait aussi laisser la fonction générer les w
    # fonction d'affichage
    hp.bode1(w1, mag1, phlin1, 'Example 17')

    ##FT2
    #Entrez les poles, les zeros et le gain de la FT #2
    z2 = [3+5j, 3-5j]
    p2 = [-3+5j, -3-5j]
    k2 = 1
    
    #Ajouter le code pour obtenir la représentation de la fonction de transfert sous forme de polynome
    tf2 = signal.zpk2tf(z2, p2, k2)
    w2, mag2, phlin2 = signal.bode(tf2, np.logspace(0, 2, 200))

    #Ajouter le code pour generer la carte des poles et zeros et le lieu de Bode
    #utiliser les fonctions dans helpers.py
    # hp.pzmap1(z2, p2, "Lieu de bode H2(s)")
    # hp.bode1(w2, mag2, phlin2, "H2(s)")
    print(tf2)
    #Question B
    
    #Analysez bien les lignes suivantes pour comprendre leur fonctionnement
    # définit les entrées pour la simulation temporelle, i.e. quelques sinusoîdes de fréquence différentes
    t = np.linspace(0, 10, 5000)  # 5000 points de 0 à 10s
    w = [0, 1, 4, 15, 7853]   # valeurs des fréquences désirées, rad/s
    # génère une constante si w = 0 sinon un sinus
    u = [np.ones_like(t) if w[i] == 0 else 0.25*np.sin(w[i] * t) for i in range(len(w))]

    # exemple réponse temporelle pour plusieurs entrées sinusoidales définies ci-dessus
    # initialise les listes qui vont contenir l'information retournée par lsim
    tout = []
    yout1 = []
    yout2 = []
    # itère sur les fréquences désirées
    for i in range(len(w)):
        #Commentez la ligne ci-dessous et decommentez les lignes commentees de la boucle for en completant les argument des appels a lsim
        #temp = [t, u[i]]
        
        temp = signal.lsim(tf1, u[i], t)  # temp = [t, y, x], voir l'aide de lsim
        tout.append(temp[0])
        yout1.append(temp[1])
        
        temp = signal.lsim(tf2, u[i], t)  # répète pour l'autre FT
        yout2.append(temp[1])
        
    #Affichage des simulations
    # hp.timepltmulti1(t, u, w, tout, yout1, 'H1')
    # hp.timepltmulti1(t, u, w, tout, yout2, 'H2')

    #Question C 
    #Comment mettre les filtres en serie/parrallele? Reflechisser a deux approches possibles, l'une utilisant lsim et l'autre utilisant les fonctions paratf et seriestf de la librairie helper
    #Ensuite, vous pouvez utiliser le meme genre de boucle que ci-dessous pour generer les reponses aux differentes frequences

    youts = []
    youtp = []

    # Serie (produit)
    zs, ps, ks = hp.seriestf(z1, p1, k1, z2, p2, k2)
    tfs = signal.zpk2tf(zs, ps, ks)
    for i in range(len(w)):
        temp = signal.lsim(tfs, u[i], t)
        youts.append(temp[1])
    hp.timepltmulti1(t, u, w, tout, youts, 'H1*H2')

    zp, pp, kp = hp.paratf(z1, p1, k1, z2, p2, k2)
    tfp = signal.zpk2tf(zp, pp, kp)
    # Parrallele (somme)
    for i in range(len(w)):
        temp = signal.lsim(tfp, u[i], t)
        youtp.append(temp[1])
    hp.timepltmulti1(t, u, w, tout, youtp, 'H1+H2')

    #Question D
    #Inspirez-vous de la question a pour afficher la carte des poles et zeros et le lieu de Bode de ces filtres, afficher aussi le delais de groupe en utilisant hp.grpdel1 en vous inspirant de la fonction exempleButterworth
    mag, ph, w, fig, ax = hp.bodeplot(tfs[0], tfs[1], 'Hs(s)')
    delay = - np.diff(ph) / np.diff(w)    # calcul
    hp.grpdel1(w, delay, 'Hs(s)')    # affichage

    mag, ph, w, fig, ax = hp.bodeplot(tfp[0], tfp[1], 'Hp(s)')
    delay = - np.diff(ph) / np.diff(w)    # calcul
    hp.grpdel1(w, delay, 'Hp(s)')    # affichage

###############################################################################
def probleme2():
    """
    Framework pour le problème 2

    :return:
    """
    
    #Question A
    
    #Frequence de coupure des filtres, a ajuster
    wc = 2 * np.pi * 2500      # fréquence de coupure rad/s
    order = 2
    b1, a1 = signal.butter(order, wc, btype='low', analog=True)
    b2, a2 = signal.butter(order, wc, btype='high', analog=True)


    #Generez les filtres butterworth avec la fonction signal.butter, sous quel forme est la sortie (polynome ou pole-zero-gain)?
    #Transformer la sortie de signal.butter dans l'autre representation (vous pouvez vous inspirer de la question A du probleme 1)


    #Affichage des polynomes
    print(f'Passe-bas Numérateur {b1}, Dénominateur {a1}')  # affiche les coefficients correspondants au filtre
    print(f'Passe-haut Numérateur {b2}, Dénominateur {a2}')  # affiche les coefficients correspondants au filtre


    #Questions B et C

    #Génère une onde carrée, ajuster la frequence
    fsquare = 1000  # Hz
    t, step = np.linspace(0, .01, 5000, retstep=True)
    u1 = signal.square(2 * np.pi * fsquare * t, 0.5)

    #Gain de chacune des bandes, ajuster au besoin
    lowk = 1
    highk = 0.7864

    #En vous inspirant de la question C du probleme 1, mettre ces deux filtres en parallele, afficher le lieu de bode et le délai de groupe
    z1 = np.roots(b1)
    p1 = np.roots(a1)
    z2 = np.roots(b2)
    p2 = np.roots(a2)
    z3, p3, k3 = hp.paratf(z1, p1, lowk, z2, p2, highk)
    tf = signal.zpk2tf(z3, p3, k3)

    #Simuler la sortie de chacun des filtres (passe-haut, passe-bas et les deux en parallele) avec la fonction lsim
    yout = []
    tf1 = (b1, a1)
    tf2 = (b2, a2)

    t = np.linspace(0, 0.01, 5000)
    temp = signal.lsim(tf1, u1, t)
    tout = temp[0]
    yout.append(temp[1])

    temp = signal.lsim(tf2, u1, t)
    yout.append(temp[1])

    temp = signal.lsim(tf, u1, t)
    yout.append(temp[1])

    systems = ("TF1", "TF2", "TF_PARA")
    hp.timepltmulti2(t, u1, tout, yout, "#2", systems)

    #Question D
    
    # génère une entrée sinusoïdale à la même fréquence que l'onde carrée
    fsin = fsquare
    u2 = np.sin(2*np.pi*fsin*t)

    # redéfinit les gains de bande
    lowk = .1
    highk = 1

    #Simuler la sortie de chacun des filtres (passe-haut, passe-bas et les deux en parallele) avec la fonction lsim et l'entre u2
    
    #Question E
    #Inspirer vous du probleme 1 pour afficher le lieu de bode et le delais de groupe des filtres conçus

def BodePasseBasCorrige():
    b1 = [1, 0, 0]  # définition du numérateur de la fonction de transfert
    a1 = [1, 1, 0.5]  # définition du dénominateur de la fonction de transfert

    # méthode 1 avec bode
    tf1 = signal.TransferFunction(b1, a1)  # définit la fonction de transfert
    # calcul le diagrame de Bode, la magnitude en dB et la phase en degrés la fréquence en rad/s
    w1, mag1, phlin1 = signal.bode(tf1, np.logspace(-1.5, 1, 200))  # on pourrait aussi laisser la fonction générer les w
    # fonction d'affichage
    hp.bode1(w1, mag1, phlin1, 'Example 1')


###############################################################################
def main():
    # décommentez la ou les lignes correspondant aux fonctions que vous voulez exécuter
    #exampleRacines()
    #exampleBode()
    #exampleButterworth()
    #probleme1()
    #probleme2()
    FiltreApp()
    #BodePasseBasCorrige()
    plt.show()
#####################################
# Manière standardisée d'exécuter ce que vous désirez lorsque vous "exécutez" le fichier
# permet d'importer les fonctions définies ici comme un module dans un autre fichier .py
# voir par exemple le fichier helper.py qui n'exécutera jamais rien tout seul
# et dont les fonctions sont importées ici à titre de support pour ce qu'on essaie de faire
# pour l'instant ne rien modifier ci-dessous
if __name__ == '__main__':
    main()
