from __future__ import annotations

import numpy as np

def hsami_etp(pas,nb_pas,jj,t_min,t_max,modules,physio):
    '''
     Calcul de l'évapotranspiration potentielle 
    
     ENTRÉES
     -------
     pas:		Pas de temps courant à l'intérieur de la journée (entier entre 1 et nb_pas)
     nb_pas:	Voir hsami2_noyau (projet['nb_pas_par_jour'])
     jj :       Jour julien (entier positif)
     t_min:     Température minimale de référence (utiliser de préférence une estimation quotidienne)
     t_max:	    Température maximale de référence (utiliser de préférence une estimation quotidienne)
     modules:   Chaine de caractère spécifiant la formulation d'ETP à utiliser
     physio:    Voir hsami2_noyau.py (projet['physio'])
    
     SORTIES
     -------
     etp:		Estimation de l'évapotranspiration potentielle
     
     MODULES D'ÉVAPOTRANSPIRATION DISPONIBLES
     ------------------
        - hsami 
        - blaney_criddle
        - hamon
        - linacre
        - kharrufa
        - mohyse
        - romanenko
        - makkink
        - turc
        - mcguinness_bordne
        - abtew
        - hargreaves
        - priestley_taylor
        
     
     EXEMPLE 
     -------
     Pas de temps de projet_test.json
     Remplacer modules['etp_bassin'] dans le projet par chacun des modules à tester 
     Les autres modules sont ceux PAR DÉFAUT.
    
    Marie Minville, Catherine Guay, 2013
    Didier Haguma, 2024    
    '''
    
    # Poids horaires pour distribuer l'évapotranspiration potentielle
    poids = np.array([.5, .5, .5, .5, .5, .6, 1.1, 2.4, 4, 5.4, 7, 8.4, 9.6, 10.4, 10.9, 10.8, 9.9, 
                    7.8, 5, 2, .7, .5, .5, .5])/100
    
    # Calcul de l'ETP total pour la journée
    
    if modules == 'hsami':                  # Ex. : etp_total = 0.1788
        etp_total = 0.00065*2.54* 9/5*(t_max-t_min) * np.exp(0.019 * (t_min*9/5 + t_max*9/5 + 64))
        
    elif modules == 'blaney_criddle':       # Ex. : etp_total = 0.2799
        p = etp_p(physio['latitude'], jj)
        etp_total = etp_blaney_criddle(t_min, t_max, p)
        
    elif modules == 'hamon':                 # Ex. : etp_total = 0.1281
        etp_total = etp_hamon(jj, t_min, t_max, physio)
        
    elif modules == 'linacre':              # Ex. : etp_total = 0.1043
        etp_total = etp_linacre(t_min, t_max, physio)
        
    elif modules == 'kharrufa':             # Ex. : etp_total = 0.0757
        p = etp_p(physio['latitude'], jj)
        etp_total = etp_kharrufa(t_min, t_max, p)
        
    elif modules == 'mohyse':               # Ex. : etp_total = 0.0812
        delta = etp_declinaison(jj)
        etp_total = etp_mohyse(t_min, t_max, delta, physio)
        
    elif modules == 'romanenko':             # Ex. : etp_total = 0.2357
        etp_total = etp_romanenko(t_min, t_max)
        
    elif modules == 'makkink':              # Ex. : etp_total = 0.2526
        Re = etp_rayonnement_et(physio['latitude'], jj)
        Rg = etp_rayonnement_g(Re, physio['latitude'], jj, t_min, t_max)
        m = etp_m_courbe_pression(t_min, t_max)
        lamda = etp_chaleur_lat_vaporisation(t_min, t_max)
        etp_total = etp_makkink(Rg, m, lamda)
        
    elif modules == 'turc':                  # Ex. : etp_total = 0.1988
        Re = etp_rayonnement_et(physio['latitude'], jj)
        Rg = etp_rayonnement_g(Re, physio['latitude'], jj, t_min, t_max)
        etp_total = etp_turc(t_min, t_max, Rg)
        
    elif modules == 'mcguinness_bordne':        #Ex. : etp_total = 0.1274
        Re = etp_rayonnement_et(physio['latitude'], jj)
        Rg = etp_rayonnement_g(Re, physio['latitude'], jj, t_min, t_max)
        lamda = etp_chaleur_lat_vaporisation(t_min, t_max)
        etp_total = etp_mcguinness_bordne(t_min, t_max, Rg, lamda)
        
    elif modules == 'abtew':                     #Ex. : etp_total = 0.4884
        Re = etp_rayonnement_et(physio['latitude'], jj)
        Rg = etp_rayonnement_g(Re, physio['latitude'], jj, t_min, t_max)
        lamda = etp_chaleur_lat_vaporisation(t_min, t_max)
        etp_total = etp_abtew(t_min, t_max, Rg, lamda)

    elif modules == 'hargreaves':                # Ex. : etp_total = 0.2566
        Re = etp_rayonnement_et(physio['latitude'], jj)
        etp_total = etp_hargreaves(t_min, t_max, Re)
        
    elif modules == 'priestley_taylor':         # Ex. : etp_total = 0.0339
        Re = etp_rayonnement_et(physio['latitude'], jj)
        Rgo = etp_rayonnement_temps_clair(Re, physio['altitude'])
        Rg = etp_rayonnement_g(Re, physio['latitude'], jj, t_min, t_max)
        Rn = etp_rayonnement_net(t_min, t_max, Rg, Rgo, physio['albedo_sol'])
        m = etp_m_courbe_pression(t_min, t_max)
        lamda = etp_chaleur_lat_vaporisation(t_min, t_max)
        etp_total = etp_priestley_taylor(Rn, m, lamda)    
        
    etp_total = max(0,etp_total)

    # Aggrégation selon le pas de temps
    debut = int((pas-1)*24/nb_pas) 
    fin   = int(pas*24/nb_pas)
    etp   = etp_total* np.sum(poids[debut:fin])

    return etp

## -----------------------------
## FIN DE LA FONCTION PRINCIPALE
## -----------------------------

## ----------------------------
## FONCTIONS DE CALCUL DE L'ETP
## ----------------------------

def etp_blaney_criddle(t_min,t_max,p):
    '''
     Calcul de l'évapotranspiration potentielle en cm
     à partir de la formulation de Blaney-Criddle
     et répartition dans la journée selon la pondération proposée par Fortin, J.P. et Girard, G. (1970)
    
     Entrees
     -------
     pas:		pas de temps courant à l'intérieur de la journée (entier entre 1 et nb_pas)
     nb_pas:	nombre de pas de temps dans une période de 24h (entier positif)
     t_min:	température minimale de référence (utiliser de préférence une estimation quotidienne)
     t_max:	température maximale de référence (utiliser de préférence une estimation quotidienne)
     p:        heures de clarté journalière sur le nombre d'heures de clarté annuelle
     

     Sortie
     ------
     etp:		estimation de l'evapotranspiration potentielle (cm)
     ==================================================================================================
    '''
    ## température moyenne
    Ta = (t_min+t_max)/2
    
    # Calcul de l'ETP total pour la journée
    k = 0.85  # Constante proposée par Xu et Singh (2001). Peut varier entre 0.5 et 1.2;
    etp_total = k*p*(0.46*Ta+8.13)/10 # cm, formulation en mm selon Xu et Singh (2001)
    etp_total = max(0,etp_total)
    
    return etp_total


def etp_hamon(jj,t_min,t_max,physio):
    '''
     Calcul de l'évapotranspiration potentielle en cm
     selon la formulation de Hamon
     et répartition dans la journée selon la pondération proposée par Fortin, J.P. et Girard, G. (1970)
    
     Entrees
     -------
     pas:		pas de temps courant à l'interieur de la journée (entier entre 1 et nb_pas)
     nb_pas:	nombre de pas de temps dans une période de 24h (entier positif)
     t_min:	température minimale de référence (utiliser de préférence une estimation quotidienne)
     t_max:	température maximale de référence (utiliser de préférence une estimation quotidienne)
     DL:       Durée d'ensoleillement journalière selon la latitude (heures)
     

     Sortie
     ------
     etp:		estimation de l'évapotranspiration potentielle (cm)
     ==================================================================================================
    '''
    DL = etp_duree_jour(jj,physio['latitude'])

    # température moyenne
    Ta = (t_min+t_max)/2

    # Pression de vapeur  
    es = etp_e(Ta)  # En émettant l'hypothése que Ts = Ta, Oudin (2004).

    # Calcul de l'ETP total pour la journée
    etp_total = 2.1 * DL**2 * es / (Ta+273.3)/10  # Haith et Shoemaker (1987). 
    etp_total = max(0,etp_total)

    return etp_total


def etp_linacre(t_min,t_max,physio):
    '''
     Calcul de l'vaépotranspiration potentielle en cm
     selon la formulaiton Linacre
     et répartition dans la journée selon la pondération proposée par Fortin, J.P. et Girard, G. (1970)
    
     Entrees
     -------
     pas:		pas de temps courant é l'interieur de la journée (entier entre 1 et nb_pas)
     nb_pas:	nombre de pas de temps dans une période de 24h (entier positif)
     t_min:	température minimale de référence (utiliser de préférence une estimation quotidienne)
     t_max:	température maximale de référence (utiliser de préférence une estimation quotidienne)
     lat:      Latitude du centre de gravite du bassin (radians). Doit être
               convertie en degrés pour cette formulaiton.
     h: altitude moyenne du bassin (m)

     Sortie
     ------
     etp:		estimation de l'vaépotranspiration potentielle (cm)
     ==================================================================================================
    
    '''
    # température moyenne
    Ta = (t_min+t_max)/2

    # température estimée é une altitude h
    Th = Ta + 0.006*physio['altitude']

    # point de rosée
    Td = etp_td_linacre(t_max,t_min)  #Td = 0.38+t_max-0.018*t_max^2+1.4+t_min-5; #Proposition de Linacre pour estimer Td, pas applicable dans les zones trés maritimes.

    # Calcul de l'ETP total pour la journée
    if Ta<0 : #le point de rosée ne peut pasêtre calculé avec une Ta négative.
        etp_total = 0
    else:
        # la latitude doitêtre en degré pour cette formulation
        lat = physio['latitude']*180/np.pi
        etp_total = (500*Th/(100-lat)+15*(Ta-Td))/((80-Ta))/10  # cm; Xu et Singh (2001)
    

    return etp_total


def etp_kharrufa(t_min,t_max,p):
    '''
     Calcul de l'vaépotranspiration potentielle en cm
     selon la formulation Kharrufa
     et répartition dans la journée selon la pondération proposée par Fortin, J.P. et Girard, G. (1970)
    
     Entrées
     -------
     pas:		pas de temps courant é l'interieur de la journée (entier entre 1 et nb_pas)
     nb_pas:	nombre de pas de temps dans une période de 24h (entier positif)
     t_min:	température minimale de référence (utiliser de préférence une estimation quotidienne)
     t_max:	température maximale de référence (utiliser de préférence une estimation quotidienne)
     p:        Heures de clarté journalière sur le nombre déheures de clarté annuelle
     

     Sortie
     ------
     etp:		estimation de l'vaépotranspiration potentielle (cm)
     ==================================================================================================
    '''
    # température moyenne
    Ta = (t_min+t_max)/2

    Ta = max(0,Ta)  # MM20130712: Ta = 0 si elle est negative car sinon ETP = nbr imaginaire
    etp_total = 0.34*p*Ta**(1.3)/10 #cm #formulation originale en mm. Xu et Singh (2001)

    return etp_total


def etp_mohyse(t_min,t_max,delta,physio):
    '''
     Calcul de l'vaépotranspiration potentielle en cm
     selon la formulation du modéle Mohyse
     et répartition dans la journée selon la pondération proposée par Fortin, J.P. et Girard, G. (1970)
    
     Entrées
     -------
     pas:		pas de temps courant é l'interieur de la journée (entier entre 1 et nb_pas)
     nb_pas:	nombre de pas de temps dans une période de 24h (entier positif)
     t_min:	température minimale de référence (utiliser de préférence une estimation quotidienne)
     t_max:	température maximale de référence (utiliser de préférence une estimation quotidienne)
     lat:      Latitude du centre de gravite du bassin (radians)
     delta:    Déclinaison solaire (radians)

     Sortie
     ------
     etp:		estimation de l'vaépotranspiration potentielle (cm)
     ==================================================================================================
     
    '''
    # température moyenne
    Ta = (t_min+t_max)/2

    # Calcul de l'ETP total pour la journée
    etp_total = 1/np.pi*np.arccos(-np.tan(physio['latitude'])*np.tan(delta))*np.exp((17.3*Ta)/(238+Ta))/10  #cm #Fortin et Turcotte (2007)

    return etp_total


def etp_romanenko(t_min,t_max):
    '''
     Calcul de l'vaépotranspiration potentielle en cm
     selon la formulation Romanenko
     et répartition dans la journée selon la pondération proposée par Fortin, J.P. et Girard, G. (1970)
    
     Entrees
     -------
     pas:		pas de temps courant é l'interieur de la journée (entier entre 1 et nb_pas)
     nb_pas:	nombre de pas de temps dans une période de 24h (entier positif)
     t_min:	température minimale de référence (utiliser de préférence une estimation quotidienne)
     t_max:	température maximale de référence (utiliser de préférence une estimation quotidienne)
     DL:        Durée d'ensoleillement journalière selon la latitude (heures)
     

     Sortie
     ------
     etp:		estimation de l'vaépotranspiration potentielle (cm)
     ==================================================================================================
    
    '''
    # température moyenne
    Ta = (t_min+t_max)/2

    #Pression de vapeur 
    ea = etp_e(Ta) 
    ed = etp_e(t_min) # on peut supposer que td=tmin pour les emplacements ou le couvert vegetal est bien humidifie (Kimball et al. (1997)

    # Calcul de l'ETP total pour la journée
    etp_total = 0.0045*(1+Ta/25)**2*(1-ed/ea)*100 #cm #La version initiale est en m. Oudin (2004)

    return etp_total


def etp_makkink(Rg,m,lamda):
    '''
     Calcul de l'vaépotranspiration potentielle en cm
     selon la formulation de Makkink
     et répartition dans la journée selon la pondération proposée par Fortin, J.P. et Girard, G. (1970)
    
     Entrees
     -------
     pas:		pas de temps courant é l'interieur de la journée (entier entre 1 et nb_pas)
     nb_pas:	nombre de pas de temps dans une période de 24h (entier positif)
     t_min:	température minimale de référence (utiliser de préférence une estimation quotidienne)
     t_max:	température maximale de référence (utiliser de préférence une estimation quotidienne)
     Rg:       Rayonnement global (MJ/m^2/j);
     m:        pente de la courbe de pression (kPa/éC);
     lamda:    Chaleur de vaporisation (MJ/kg);


     Sortie
     ------
     etp:		estimation de l'vaépotranspiration potentielle (cm)
     ==================================================================================================
    
    '''

    psi = 0.066 # Constante psychométrique (0,066 kPa/éC); 

    # Calcul de l'ETP total pour la journée
    etp_total = ((m/(m+psi))*(0.61*Rg/lamda)-.12)/10 # http://books.google.ca/books?id=oqYoAAAAYAAJ&pg=PA33&lpg=PA33&dq=Makkink+evapotranspiration+0.61+0.12&source=bl&ots=xuCbxh-Shc&sig=1CKhFyAEDm4ufjmPRhMwL-n1LkE&hl=fr&sa=X&ei=A8MMUoSxDaT4yQHYoYGQCA&ved=0CDwQ6AEwAQ#v=onepage&q=turc&f=false #http://www.luiw.ethz.ch/labor2/experimente/exp4/Presentation/Empirical_ET_Models

    return etp_total


def etp_turc(t_min,t_max,Rg):
    '''
     Calcul de l'vaépotranspiration potentielle en cm
     à partir des températures min et max en C journalière
     selon la méthode empirique de Jean-Louis Bisson (Hydro-Québec)
     et répartition dans la journée selon la pondération proposée par Fortin, J.P. et Girard, G. (1970)
    
     Entrées
     -------
     pas:		pas de temps courant é l'interieur de la journée (entier entre 1 et nb_pas)
     nb_pas:	nombre de pas de temps dans une période de 24h (entier positif)
     t_min:	température minimale de référence (utiliser de préférence une estimation quotidienne)
     t_max:	température maximale de référence (utiliser de préférence une estimation quotidienne)
     Rg:       Rayonnement global (MJ/m^2/j);
     lamda:    Chaleur de vaporisation (MJ/kg);

     Sortie
     ------
     etp:		estimation de l'vaépotranspiration potentielle (cm)
     ==================================================================================================
    
    '''

    # température moyenne
    Ta = (t_min+t_max)/2

    K = 0.35 #Constante de Turc

    # Calcul de l'ETP total pour la journée
    if Ta<0:
        etp_total=0
    else :
        etp_total = K*(Rg+2.094)*(Ta/(Ta+15))/10 #cm; McGuiness et Bordne (1972), unité mise en SI tel que documenté dans http://www.luiw.ethz.ch/labor2/experimente/exp4/Presentation/Empirical_ET_Models
    

    return etp_total


def etp_mcguinness_bordne(t_min,t_max,Rg,lamda):
    '''
    
     Calcul de l'vaépotranspiration potentielle en cm
     selon la formulation de McGuiness et Bordne
     et répartition dans la journée selon la pondération proposée par Fortin, J.P. et Girard, G. (1970)
    
     Entrées
     -------
     pas:		pas de temps courant é l'interieur de la journée (entier entre 1 et nb_pas)
     nb_pas:	nombre de pas de temps dans une période de 24h (entier positif)
     t_min:	température minimale de référence (utiliser de préférence une estimation quotidienne)
     t_max:	température maximale de référence (utiliser de préférence une estimation quotidienne)
     Rg:       Rayonnement global (MJ/m^2/j);
     lamda:    Chaleur de vaporisation (MJ/kg);

     Sortie
     ------
     etp:		estimation de l'vaépotranspiration potentielle (cm)
     ==================================================================================================
    
    '''

    # température moyenne
    Ta = (t_min+t_max)/2

    rho_w = 1000 # Masse volumique de léeau (1000 kg/m3)

    # Calcul de l'ETP total pour la journée
    
    etp_total = (Rg/(lamda*rho_w)*(Ta+5)/68)*100 # cm, version originale en m. Oudin (2004)

    return etp_total


def etp_abtew(t_min,t_max,Rg,lamda):
    '''
         
     Calcul de l'évapotranspiration potentielle en cm
     selon la méthode empirique de Abtew
     et répartition dans la journée selon la pondération proposée par Fortin, J.P. et Girard, G. (1970)
    
     Entrées
     -------
     pas:		pas de temps courant é l'interieur de la journée (entier entre 1 et nb_pas)
     nb_pas:	nombre de pas de temps dans une période de 24h (entier positif)
     t_min:	température minimale de référence (utiliser de préférence une estimation quotidienne)
     t_max:	température maximale de référence (utiliser de préférence une estimation quotidienne)
     Rg:       Rayonnement global MJ/m^2/j;
     lamda:    Chaleur de vaporisation MJ/kg;

     Sortie
     ------
     etp:		estimation de l'évapotranspiration potentielle (cm)
    
     ==================================================================================================
    
    '''
    # température moyenne
    Ta = (t_min+t_max)/2

    # Calcul de l'ETP total pour la journée
    if Ta<0:
        etp_total=0
    else:
        etp_total = 0.53*Rg/lamda/10 #Xu et Singh 2010

    return etp_total


def etp_hargreaves(t_min,t_max,Re):
    '''
     
     Calcul de l'vaépotranspiration potentielle en cm
     selon la formulation de Hargreaves et Samani
     et répartition dans la journée selon la pondération proposée par Fortin, J.P. et Girard, G. (1970)
    
     Entrées
     -------
     pas:		pas de temps courant é l'interieur de la journée (entier entre 1 et nb_pas)
     nb_pas:	nombre de pas de temps dans une période de 24h (entier positif)
     t_min:	température minimale de référence (utiliser de préférence une estimation quotidienne)
     t_max:	température maximale de référence (utiliser de préférence une estimation quotidienne)
     Re:       Rayonnement extraterrestre (MJ/m^2/j)


     Sortie
     ------
     etp:		estimation de l'vaépotranspiration potentielle (cm)
     ==================================================================================================
    
    
    '''
    # température moyenne
    Ta = (t_min+t_max)/2

    # Calcul de l'ETP total pour la journée
    if t_max-t_min < 0 :#il y a parfois des incohérence dans les séries observées. Cette condition pourraitêtre enlevée éventuellement.
        etp_total = 0
    else:
        etp_total = 0.0135*(0.16*Re*np.sqrt(t_max-t_min))*0.4082*(Ta+17.8)/10 # Goyal et Harmsen (2014). Extrait du livre via Google book.


    return etp_total


def etp_priestley_taylor(Rn,m,lamda):
    '''
     
     Calcul de l'vaépotranspiration potentielle en cm
     selon la formulation de Priesley-Taylor
     et répartition dans la journée selon la pondération proposée par Fortin, J.P. et Girard, G. (1970)
    
     Entrees
     -------
     pas:		pas de temps courant é l'interieur de la journée (entier entre 1 et nb_pas)
     nb_pas:	nombre de pas de temps dans une période de 24h (entier positif)
     t_min:	température minimale de référence (utiliser de préférence une estimation quotidienne)
     t_max:	température maximale de référence (utiliser de préférence une estimation quotidienne)
     Rn:       Rayonnement net (MJ/m^2/j)
     m:        pente de la courbe de pression 
     lamda:    Chaleur de vaporisation (MJ/kg)

     Sortie
     ------
     etp:		estimation de l'vaépotranspiration potentielle (cm)
     ==================================================================================================
    
    '''

    psi   = 0.066    # Constante psychométrique (0,066 kPa/éC); 
    rho_w = 1000     # Masse volumique de l'eau (kg/m3)

    ct = 1.26        #constante proposée par Priesley-Taylor

    # Calcul de l'ETP total pour la journée
    etp_total = ct*m*Rn/(lamda*rho_w*(m+psi))*100 #cm; La formule proposée est en m. Oudin (2004)

    return etp_total


# --------------------
# FONCTIONS DE SOUTIEN
# --------------------

def etp_p(lat,jj):
    '''

     Calcul du pourcentage de la durée du jour sur la somme des durées du jour
     annuelles

     Entrees
     -------
     lat:   latitude moyenne du bassin versant
     jj:   jour julien

     Sortie
     ------
     p :    Heures de clarté journalière sur le nombre d'heures de clarté annuelle
     =========================================================================
    
    '''
    DL = np.zeros(366)

    for jj2 in range(366):
        DL[jj2] = etp_duree_jour(jj2,lat)
    

    p = 100*(DL[jj]/np.sum(DL))    #Xu et Singh (2000)

    return p


def etp_duree_jour(jj,lat):
    '''
     
     Calcul de la duree du jour jj à la latitude lat.

     Entrees
     -------
     lat:   latitude moyenne du bassin versant
     jj:   jour julien

     Sortie
     ------
     DL:    Duree du jour au jour jj et à la latitude lat
     =========================================================================
    
    '''
    delta = etp_declinaison(jj)

    #http://www.argenco.ulg.ac.be/etudiants/Multiphysics/Xanthoulis#20-#20Calcul#20ETo#20-#20Penman.pdf
    ws = np.arccos(-np.tan(lat)*np.tan(delta))    #angle de coucher de soleil (rad) 
    DL = 24/np.pi*ws

    return DL


def etp_declinaison(jj):
    '''
    
     Calcul de la declinaison solaire (en radians) au jour julien jj

     Entrees
     -------
     jj:   jour julien

     Sortie
     ------
     delta:    Declinaison du soleil (radians) pour mohyse entre autre
     =========================================================================
    
    '''
    delta = 0.41 * np.sin((jj-80)/365*2*np.pi)

    return delta

def etp_td_linacre(t_max,t_min):
    '''
     Estimation du point de rosée de Linacre.

     Entrees
     -------
     t_max:   Tmax journalière
     t_min:   Tmin journalière 

     Sortie
     ------ 
     Td:    Point de rosée
     =========================================================================
    
    '''
    # Td Point de rosée
    Td = 0.38+t_max-0.018*t_max**2+1.4+t_min-5  #Proposition de Linacre pour estimer Td, pas applicable dans les zones trés maritimes.
    
    return Td

def etp_rayonnement_et(lat,jj):
    '''
     
     Calcul du rayonnement extra-terrestre.

     Entrees
     -------
     lat:   latitude moyenne du bassin versant
     jj:   jour julien

     Sortie
     ------
     Re:    Rayonnement extra-terrestre (MJ/m^2/j)
     =========================================================================

     Selon
     http://www.argenco.ulg.ac.be/etudiants/Multiphysics/Xanthoulis#20-#20Calcul#20ETo#20-#20Penman.pdf
    
    '''
    Gsc = 0.0820 #MJ/m2/j constante solaire

    dr = 1+0.033*np.cos(2*np.pi/365*jj)         #distance relative inverse terre-soleil (rad)
    delta = 0.409*np.sin(2*np.pi*jj/365-1.39)   #declinaison solaire (rad)
    ws = np.arccos(-np.tan(lat)*np.tan(delta))    #angle de coucher de soleil (rad) 

    Re = 24*60/np.pi*Gsc*dr*((ws*np.sin(lat)*np.sin(delta)+(np.cos(lat)*np.cos(delta)*np.sin(ws))))

    return Re


def etp_rayonnement_g(Re,lat,jj,t_min=None,t_max=None):
    '''
     
     Calcul du rayonnement global.

     Entrees
     -------
     Re:       Rayonnement extra-terrestre (MJ/m^2/j)
     lat:      Latitude moyenne du bassin versant (m)
     jj:       Jour julien
     t_min:    Tmin journalière
     t_max:    Tmax journalière

     Sortie
     ------
     Rg:   Rayonnement global (MJ/m^2/j)
     =========================================================================
    
    '''

    DL = etp_duree_jour(jj,lat)
    D = 0.8*DL # Hypothése, nous n'avons pas d'observation pour estimer la durée effective du jour.

    #Rayonnement global
    Rg = Re*(0.18+0.52*D/DL)

    # Autre facon, bien si on ne connait pas D.
    if t_min != None:
        Krs=0.175
        Rg = Krs*(t_max-t_min)**(1/2)*Re
    
    # La différence entre la température maximum et minimum (Tmax-Tmin) de 
    # l'air peut être utilisé comme un indicateur de la fraction de radiation 
    # extraterrestre qui atteint la surface du sol.
    # Ra : Rayonnement extraterrestre [MJ m-2d-1],
    # Tmax: température maximum de l'air [oC],Tmin: température minimum de l'aair [oC],
    # Krs: Coefficient (0.16.. 0.19) [oC-0.5].pour des zones interieures où 
    # les masses de terres ne sont pas influencées fortement par de grandes masses d'eau: 
    # Krs= 0.16; pour des zones cétiéres situées sur ou adjacentes à une grande 
    # masse de terre et où les masses d'air sont influencées par une masse d'eau proche: Krs= 0.19.
    return Rg


def etp_m_courbe_pression(t_min,t_max):
    '''
     
     Estimation de la pente de la courbe de pression de vapeur.

     Entrees
     -------
     t_min:    Tmin journalière
     t_max:    Tmax journalière

     Sortie
     ------
     m:    Pente de la courbe de pression de vapeur kPa/degC (*10 pour avoir en mbar)
     =========================================================================
    
    '''

    # température moyenne
    Ta = (t_min+t_max)/2

    # ea pression de vapeur
    ea = etp_e(Ta)

    # m: pente
    m = 4098*ea/(237.3+Ta)**2  #Oudin (2004)

    return m

def etp_e(T):
    '''
     
     Estimation du point de la pression de vapeur.

     Entrees
     -------
     T: température

     Sortie
     ------
     e:    Pression de vapeur
     =========================================================================
    
    '''
    # e Pression de vapeur
    e = 0.6108*np.exp((17.27*T)/(T+237.3))    # Lu et al. (2005)  #pression de vapeur é T

    return e

def etp_chaleur_lat_vaporisation(t_max,t_min):
    '''
     
     Estimation de la chaleur latente de vaporisation.

     Entrees
     -------
     t_max:   Tmax journalière
     t_min:   Tmin journalière 

     Sortie
     ------
     lamda:    Chaleur latente de vaporisation (MJ/kg)
     =========================================================================
    
    '''

    # température moyenne
    Ta = (t_min+t_max)/2

    # lamda
    lamda = 2.5-2.36*10**-3*Ta      #selon Dingman, p.274. En MJ/kg. A 20degC, ca revient au flux de chaleur latente de 2.45 MJ/kg fixé dans Oudin et al. (2005)

    return lamda

def etp_rayonnement_net(t_min,t_max,Rg,Rgo,albedo):
    '''
     
     Calcul du rayonnement net.

     Entrees
     -------
     t_min:    Tmin journalière
     t_max:    Tmax journalière
     Rg:       Rayonnement global (MJ/m^2/j)
     Rgo:      Rayonnement par temps clair (MJ/m^2/j)
     albedo:   Albedo de la surface

     Sortie
     ------
     Rn:   Rayonnement net (MJ/m^2/j)
     =========================================================================
    
    '''
    #Rayonnement net de courte longueur d'ondes
    Rns = Rg*(1-albedo)

    #Rayonnement net de longue longueur d'ondes
    sigma = 4.903*10**(-9)  # constante de S-B
    K = 273.16 #pour avoir des Kelvins

    ea = etp_e(t_min)   #Td = Tmin est une approximation valable (Kimball et al. 1997)

    rapport = Rg/Rgo
    if rapport >= 1:
        rapport=1       #selon Xu et Singh (2002) - WRM
    
    Rnl = sigma*((t_max+K)**4+(t_min+K)**4)/2*(0.34-0.14*np.sqrt(ea))*(1.35*rapport-0.35)


    #Rayonnement net
    Rn = Rns - Rnl

    return Rn

def etp_rayonnement_temps_clair(Re,h):
    '''
    
     Calcul du rayonnement par temps clair.

     Entrees
     -------
     Re:   Rayonnement extraterrestre (MJ/m^2/j)
     h:   Hauteur moyenne du bassin versant au dessus du niveau de la mer (m)

     Sortie
     ------
     Rgo:    Rayonnement pas temps clair considérant D=DL (MJ/m^2/j)
     =========================================================================
    
    '''

    #Rayonnement solaire par temps clair.
    Rgo = (0.75+2.10*10**-5*h)*Re   #Xu et Singh (2002). WRM

    return Rgo
