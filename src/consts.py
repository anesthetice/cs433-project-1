# fmt: off
IDX_TO_COLNAME: list[str] = ['_STATE', 'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'DISPCODE', 'SEQNO', '_PSU', 'CTELENUM', 'PVTRESD1', 'COLGHOUS', 'STATERES', 'CELLFON3', 'LADULT', 'NUMADULT', 'NUMMEN', 'NUMWOMEN', 'CTELNUM1', 'CELLFON2', 'CADULT', 'PVTRESD2', 'CCLGHOUS', 'CSTATE', 'LANDLINE', 'HHADULT', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'HLTHPLN1', 'PERSDOC2', 'MEDCOST', 'CHECKUP1', 'BPHIGH4', 'BPMEDS', 'BLOODCHO', 'CHOLCHK', 'TOLDHI2', 'CVDSTRK3', 'ASTHMA3', 'ASTHNOW', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD1', 'HAVARTH3', 'ADDEPEV2', 'CHCKIDNY', 'DIABETE3', 'DIABAGE2', 'SEX', 'MARITAL', 'EDUCA', 'RENTHOM1', 'NUMHHOL2', 'NUMPHON2', 'CPDEMO1', 'VETERAN3', 'EMPLOY1', 'CHILDREN', 'INCOME2', 'INTERNET', 'WEIGHT2', 'HEIGHT3', 'PREGNANT', 'QLACTLM2', 'USEEQUIP', 'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'SMOKE100', 'SMOKDAY2', 'STOPSMK2', 'LASTSMK2', 'USENOW3', 'ALCDAY5', 'AVEDRNK2', 'DRNK3GE5', 'MAXDRNKS', 'FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVGREEN', 'FVORANG', 'VEGETAB1', 'EXERANY2', 'EXRACT11', 'EXEROFT1', 'EXERHMM1', 'EXRACT21', 'EXEROFT2', 'EXERHMM2', 'STRENGTH', 'LMTJOIN3', 'ARTHDIS2', 'ARTHSOCL', 'JOINPAIN', 'SEATBELT', 'FLUSHOT6', 'FLSHTMY2', 'IMFVPLAC', 'PNEUVAC3', 'HIVTST6', 'HIVTSTD3', 'WHRTST10', 'PDIABTST', 'PREDIAB1', 'INSULIN', 'BLDSUGAR', 'FEETCHK2', 'DOCTDIAB', 'CHKHEMO3', 'FEETCHK', 'EYEEXAM', 'DIABEYE', 'DIABEDU', 'CAREGIV1', 'CRGVREL1', 'CRGVLNG1', 'CRGVHRS1', 'CRGVPRB1', 'CRGVPERS', 'CRGVHOUS', 'CRGVMST2', 'CRGVEXPT', 'VIDFCLT2', 'VIREDIF3', 'VIPRFVS2', 'VINOCRE2', 'VIEYEXM2', 'VIINSUR2', 'VICTRCT4', 'VIGLUMA2', 'VIMACDG2', 'CIMEMLOS', 'CDHOUSE', 'CDASSIST', 'CDHELP', 'CDSOCIAL', 'CDDISCUS', 'WTCHSALT', 'LONGWTCH', 'DRADVISE', 'ASTHMAGE', 'ASATTACK', 'ASERVIST', 'ASDRVIST', 'ASRCHKUP', 'ASACTLIM', 'ASYMPTOM', 'ASNOSLEP', 'ASTHMED3', 'ASINHALR', 'HAREHAB1', 'STREHAB1', 'CVDASPRN', 'ASPUNSAF', 'RLIVPAIN', 'RDUCHART', 'RDUCSTRK', 'ARTTODAY', 'ARTHWGT', 'ARTHEXER', 'ARTHEDU', 'TETANUS', 'HPVADVC2', 'HPVADSHT', 'SHINGLE2', 'HADMAM', 'HOWLONG', 'HADPAP2', 'LASTPAP2', 'HPVTEST', 'HPLSTTST', 'HADHYST2', 'PROFEXAM', 'LENGEXAM', 'BLDSTOOL', 'LSTBLDS3', 'HADSIGM3', 'HADSGCO1', 'LASTSIG3', 'PCPSAAD2', 'PCPSADI1', 'PCPSARE1', 'PSATEST1', 'PSATIME', 'PCPSARS1', 'PCPSADE1', 'PCDMDECN', 'SCNTMNY1', 'SCNTMEL1', 'SCNTPAID', 'SCNTWRK1', 'SCNTLPAD', 'SCNTLWK1', 'SXORIENT', 'TRNSGNDR', 'RCSGENDR', 'RCSRLTN2', 'CASTHDX2', 'CASTHNO2', 'EMTSUPRT', 'LSATISFY', 'ADPLEASR', 'ADDOWN', 'ADSLEEP', 'ADENERGY', 'ADEAT1', 'ADFAIL', 'ADTHINK', 'ADMOVE', 'MISTMNT', 'ADANXEV', 'QSTVER', 'QSTLANG', 'MSCODE', '_STSTR', '_STRWT', '_RAWRAKE', '_WT2RAKE', '_CHISPNC', '_CRACE1', '_CPRACE', '_CLLCPWT', '_DUALUSE', '_DUALCOR', '_LLCPWT', '_RFHLTH', '_HCVU651', '_RFHYPE5', '_CHOLCHK', '_RFCHOL', '_LTASTH1', '_CASTHM1', '_ASTHMS1', '_DRDXAR1', '_PRACE1', '_MRACE1', '_HISPANC', '_RACE', '_RACEG21', '_RACEGR3', '_RACE_G1', '_AGEG5YR', '_AGE65YR', '_AGE80', '_AGE_G', 'HTIN4', 'HTM4', 'WTKG3', '_BMI5', '_BMI5CAT', '_RFBMI5', '_CHLDCNT', '_EDUCAG', '_INCOMG', '_SMOKER3', '_RFSMOK3', 'DRNKANY5', 'DROCDY3_', '_RFBING5', '_DRNKWEK', '_RFDRHV5', 'FTJUDA1_', 'FRUTDA1_', 'BEANDAY_', 'GRENDAY_', 'ORNGDAY_', 'VEGEDA1_', '_MISFRTN', '_MISVEGN', '_FRTRESP', '_VEGRESP', '_FRUTSUM', '_VEGESUM', '_FRTLT1', '_VEGLT1', '_FRT16', '_VEG23', '_FRUITEX', '_VEGETEX', '_TOTINDA', 'METVL11_', 'METVL21_', 'MAXVO2_', 'FC60_', 'ACTIN11_', 'ACTIN21_', 'PADUR1_', 'PADUR2_', 'PAFREQ1_', 'PAFREQ2_', '_MINAC11', '_MINAC21', 'STRFREQ_', 'PAMISS1_', 'PAMIN11_', 'PAMIN21_', 'PA1MIN_', 'PAVIG11_', 'PAVIG21_', 'PA1VIGM_', '_PACAT1', '_PAINDX1', '_PA150R2', '_PA300R2', '_PA30021', '_PASTRNG', '_PAREC1', '_PASTAE1', '_LMTACT1', '_LMTWRK1', '_LMTSCL1', '_RFSEAT2', '_RFSEAT3', '_FLSHOT6', '_PNEUMO2', '_AIDTST3']

COLNAME_TO_IDX: dict[str, int] = {
    name: idx for idx, name in enumerate(IDX_TO_COLNAME)
}

# NaN will always be treated as an unknown value.
INTERESTING_FEATURES: list[dict] = [

    # General Health
    dict(colname="GENHLTH", type="ordcat", uvs=[7, 9], rnw=7),
    # Number of Days Physical Health Not Good
    dict(colname="PHYSHLTH", type="numerical", uvs=[77, 99], rnw=77),
    # Have any health care coverage
    dict(colname="HLTHPLN1", type="ordcat", uvs=[7, 9], rnw=7),
    # Multiple Health Care Professionals (not ordered categorical since values are not ordered: 1, 2, 0)
    dict(colname="PERSDOC2", type="cat", uvs=[7, 9]),
    # Could Not See Doctor Because of Cost
    dict(colname="MEDCOST", type="ordcat", uvs=[7, 9], rnw=7),
    # Length of time since last routine checkup
    dict(colname="CHECKUP1", type="ordcat", uvs=[7, 9], rnw=7, replace=[(8, 5)]),

    # Calculated variable for adults who have been told they have high blood pressure
    dict(colname="_RFHYPE5", type="ordcat", uvs=[9], rnw=9),
    # Currently Taking Blood Pressure Medication
    dict(colname="BPMEDS", type="ordcat", uvs=[7, 9], rnw=7),

    # How Long since Cholesterol Checked
    dict(colname="CHOLCHK", type="ordcat", uvs=[7, 9], rnw=7),
    # Ever Told Blood Cholesterol High
    dict(colname="TOLDHI2", type="ordcat", uvs=[7, 9], rnw=7),

    # Ever had a stroke?
    dict(colname="CVDSTRK3", type="ordcat", uvs=[7, 9], rnw=7),

    # Calculated variable for computed asthma status. (1=Current, 2=Former, 3=Never)
    dict(colname="_ASTHMS1", type="ordcat", uvs=[9], rnw=9),

    # Ever told you had skin cancer?
    dict(colname="CHCSCNCR", type="ordcat", uvs=[7, 9], rnw=7),
    # Ever told you had any other types of cancer?
    dict(colname="CHCOCNCR", type="ordcat", uvs=[7, 9], rnw=7),

    # Ever told you have chronic obstructive pulmonary disease, emphysema or chronic bronchitis?
    dict(colname="CHCCOPD1", type="ordcat", uvs=[7, 9], rnw=7),

    # Told Have Arthritis
    dict(colname="HAVARTH3", type="ordcat", uvs=[7, 9], rnw=7),

    # Ever told you had a depressive disorder
    dict(colname="ADDEPEV2", type="ordcat", uvs=[7, 9], rnw=7),

    # Ever told you have kidney disease?
    dict(colname="CHCKIDNY", type="ordcat", uvs=[7, 9], rnw=7),

    # Ever told you have diabetes
    dict(colname="DIABETE3", type="ordcat", uvs=[7, 9], rnw=7, replace=[(3, 5)]),

    # ---

    # Respondents Sex: male = 1, female = 2
    dict(colname="SEX", type="ordcat", uvs=[]),

    # Education Level
    dict(colname="EDUCA", type="ordcat", uvs=[9], rnw=9),

    # Own or Rent Home
    dict(colname="RENTHOM1", type="ordcat", uvs=[7, 9], rnw=7),

    # Are You A Veteran
    dict(colname="VETERAN3", type="ordcat", uvs=[7, 9], rnw=7),

    # Employment Status
    dict(colname="EMPLOY1", type="cat", uvs=[9]),

    # DO THESE MANUALLY
    # Number of Adults in Household (land line)
    #dict(colname="NUMADULT", type="numerical", uvs=[], rnw=?),
    # Number of Adults in Household (cell phone)
    #dict(colname="HHADULT", type="numerical", uvs=[77, 99], rnw=77),
    # Income Level
    dict(colname="INCOME2", type="ordcat", uvs=[77, 99], rnw=77),

    # Activity Limitation Due to Health Problems
    dict(colname="QLACTLM2", type="ordcat", uvs=[7, 9], rnw=7),
    # Health Problems Requiring Special Equipment
    dict(colname="USEEQUIP", type="ordcat", uvs=[7, 9], rnw=7),
    # Difficulty Concentrating or Remembering
    dict(colname="DECIDE", type="ordcat", uvs=[7, 9], rnw=7),
    # Difficulty Walking or Climbing Stairs
    dict(colname="DIFFWALK", type="ordcat", uvs=[7, 9], rnw=7),
    # Difficulty Dressing or Bathing
    dict(colname="DIFFDRES", type="ordcat", uvs=[7, 9], rnw=7),
    # Difficulty Doing Errands Alone
    dict(colname="DIFFALON", type="ordcat", uvs=[7, 9], rnw=7),

    # Calculated variable for "race", there's a lot of these but this one seems somewhat reasonable (still not great though, but then again this is an american survey after all...)
    dict(colname="_RACE", type="cat", uvs=[9]),
    # Calculated variable for fourteen level age category, "AGE" by itself isn't here unfortunately.
    dict(colname="_AGEG5YR", type="ordcat", uvs=[14], rnw=14),

    # Calculated variable for reported height in meters.
    dict(colname="HTM4", type="numerical", uvs=[], rnw=9.99),
    # Calculated variable for reported weight in kilograms.
    dict(colname="WTKG3", type="numerical", uvs=[], rnw=999.99),
    # Calculated variable for four - categories of body mass index (bmi).
    dict(colname="_BMI5CAT", type="ordcat", uvs=[], rnw=7),

    # ---

    # Calculated variable for four - level smoker status: everyday smoker, someday smoker, former smoker, non-smoker.
    dict(colname="_SMOKER3", type="ordcat", uvs=[9], rnw=9),

    # ---

    # Calculated variable for calculated total number of alcoholic beverages consumed per week.
    dict(colname="_DRNKWEK", type="numerical", uvs=[99900], rnw=99900),
    # Calculated variable for heavy drinkers
    dict(colname="_RFDRHV5", type="ordcat", uvs=[9], rnw=9),

    # ---

    # Calculated variable for total fruits consumed per day.
    dict(colname="_FRUTSUM", type="numerical", uvs=[], rnw=999.99),
    # Calculated variable for consume fruit 1 or more times per day.
    dict(colname="_FRTLT1", type="ordcat", uvs=[9], rnw=9),

    # Calculated variable for total vegetables consumed per day.
    dict(colname="_VEGESUM", type="numerical", uvs=[], rnw=999.88),
    # Calculated variable for consume vegetables 1 or more times per day.
    dict(colname="_VEGLT1", type="ordcat", uvs=[9], rnw=9),

    # ---

    # Calculated variable for minutes of total physical activity per week.
    dict(colname="PA1MIN_", type="numerical", uvs=[], rnw=99999),
    # Calculated variable for minutes of total vigorous physical activity per week.
    dict(colname="PA1VIGM_", type="numerical", uvs=[], rnw=99999),
    # Calculated variable for physical activity categories.
    dict(colname="_PACAT1", type="ordcat", uvs=[9], rnw=9),
    # Calculated variable for muscle strengthening recommendation.
    dict(colname="_PASTRNG", type="ordcat", uvs=[9], rnw=9),

    # ---
    # These are the more sparse modules ahead
    # ---

    # Doctor Advised Reduced Sodium/Salt Intake
    dict(colname="DRADVISE", type="ordcat", uvs=[7, 9], rnw=7),

    # Outpatient rehab after heart attack hosp stay
    # (Why didn't they remove this from the dataset though...)
    # (Note that yes + no ≃ 1000 here while for heart attack question yes ≃ 25'000)
    dict(colname="CVDASPRN", type="ordcat", uvs=[7, 9], rnw=7),

    # Take Aspirin Daily or Every Other Day
    dict(colname="CVDASPRN", type="ordcat", uvs=[7, 9], rnw=7),
    # Health Makes Taking Aspirin Unsafe
    dict(colname="ASPUNSAF", type="ordcat", uvs=[7, 9], rnw=7, replace=[(2, 1)]),
    # Take Aspirin to Relieve Pain
    dict(colname="RLIVPAIN", type="ordcat", uvs=[7, 9], rnw=7),
    # Take Aspirin to Reduce Chance of Heart Attack
    dict(colname="RDUCHART", type="ordcat", uvs=[7, 9], rnw=7),

    # Times Past 12 Months Worried/Stressed About Having Enough Money To Pay Your Rent/Mortgage?
    dict(colname="SCNTMNY1", type="ordcat", uvs=[7, 8, 9], rnw=7),
    # Times Past 12 Months Worried/Stressed About Having Enough Money To Buy Nutritious Meals?
    dict(colname="SCNTMEL1", type="ordcat", uvs=[7, 8, 9], rnw=7),
    # How Many Hours Per Week Do You Work
    dict(colname="SCNTWRK1", type="numerical", uvs=[97, 99], rnw=97, replace=[(98, 0)]),

    #dict(colname="", type="ordcat", uvs=[7, 9], rnw=7),
    #dict(colname="", type="cat", uvs=[9]),
    #dict(colname="", type="numerical", uvs=[], rnw=99999),
]
# fmt: on
