"""
Abstract
========
Fear as an emotional reaction to threatening stimuli or situations is a common feeling with high evolutionary relevance. Fear activates the fight or flight system and should assure survival. Individuals differ in the way they cope with potential threat. Krohne (Krohne, 1989) differentiated in their model of coping between two coping styles: cognitive avoidance and vigilance. Cognitive avoidance is characterized through the avoidance of threat-related information; whereas the vigilant strategy is characterized through an approach an intensive processing of threat-relevant information. Classifying individuals according to their preferred coping style (e.g. identified through the Mainz Coping Inventory, MCI, Krohne & Egloff, 1999), several studies investigated behaviour and neurobiology.  One study showed that in repressors there is a high discrepancy between self-reported and psychophysiological/ behavioral anxiety measures (Derakshan & Eysenck, 2001).

Unabhängige Daten:
 - T1: W:\NAE\SPM\2ndLevel_Lisa\ABI_N\Data\N_all_Greymatter
 - Funktionelle Kontrastbilder: W:\NAE\SPM\2ndLevel_Lisa\ABI_N\Data\N_all
    - con_10 für angry versus neutral
    - con_12 für fearful versus neutral
    - con17 für Negative (also angry und fearful gemeinsam) versus neutral
 - Zu verwendende ROIs:
    - Amygdala
    - Acc
    - Subgenaules Cingulum, BA25

Abhängige Daten:
 - W:\NAE\SPM\2ndLevel_Lisa\ABI_N\Decriptives\ 29170718_Ausgangsdatei_mit Summenscores_Brain
    - nur die included_HC_all = 1 nehmen
    - zu prädizierende Variablen:
       - ABI_Vig (= Vigilanz als Copingstrategie)
       - Abi_KVermeidung (für kognitive Vermeidung als Strategie)
"""

from Logging import Logger


def loading_t1_mris():
    # todo
    # T1: W:\NAE\SPM\2ndLevel_Lisa\ABI_N\Data\N_all_Greymatter
    Logger().info("Loading T1 MRIs")
    raise Exception("Function not implemented, needs to be done!")
    return None

def extract_mri_roi(mri_scan, roi_name):
    if roi_name = "amygdala":
        Logger().info("Extracting Amygdala")
        raise Exception("Function not implemented, needs to be done!")
        return None
    elif roi_name = "acc":
        Logger().info("Extracting ACC")
        raise Exception("Function not implemented, needs to be done!")
        return None
    elif roi_name = "gyrus_cinguli":
        Logger().info("Extracting Gyrus cinguli")
        raise Exception("Function not implemented, needs to be done!")
        return None
    else:
        raise Exception("'{}' is a not supported region of interest!".format(roi_name))

def loading_abi_variables():
    # Todo
    #  - W:\NAE\SPM\2ndLevel_Lisa\ABI_N\Decriptives\ 29170718_Ausgangsdatei_mit Summenscores_Brain
    #    - nur die included_HC_all = 1 nehmen
    #    - zu prädizierende Variablen:
    #      - ABI_Vig (= Vigilanz als Copingstrategie)
    #       - Abi_KVermeidung (für kognitive Vermeidung als Strategie)

    Logger().info("Loading ABI variables")
    raise Exception("Function not implemented, needs to be done!")
    abi_vigilanz = None
    abi_kognitive_vermeidung = None
    return (abi_vigilanz, abi_kognitive_vermeidung)

# Lade abhängige Variablen
t1_mris = loading_t1_mris()
t1_amygdala_mris = extract_mri_roi(t1_mris, "amygdala")
t1_acc_mris = extract_mri_roi(t1_mris, "acc")
t1_gyrus_cinguli_mris = extract_mri_roi(t1_mris, "gyrus_cinguli")

# Lade unabhängige Variablen
abi_vigilanz, abi_kognitive_vermeidung = loading_abi_variables()

# Building Hyperpipes
# ToDo
