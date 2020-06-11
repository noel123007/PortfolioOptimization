import os
import quandl
import numpy as np
import pandas as pd
from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    calc = np.sum(mean_returns * weights)
    returns = calc * 252
    std = np.sqrt(np.dot(weights.T, np.dot(
        cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(4)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(
            weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        # Sharpe Ratio
    return results, weights_record


# Allocation weights of based on Minimum Volaitility
alloc_min_A = 0
alloc_min_B = 0
alloc_min_C = 0
alloc_min_D = 0

# Allocation weights of based on Sharpe Ratio

alloc_sr_A = 0
alloc_sr_B = 0
alloc_sr_C = 0
alloc_sr_D = 0


def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, stocks):
    results, weights = random_portfolios(
        num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(
        weights[max_sharpe_idx], index=table.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [
        round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(
        weights[min_vol_idx], index=table.columns, columns=['allocation'])
    min_vol_allocation.allocation = [
        round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252
    # Extracting allocation weight for Minimum Volatility
    global alloc_min_A
    global alloc_min_B
    global alloc_min_C
    global alloc_min_D

    alloc_min_A = int(min_vol_allocation.at['allocation', stocks[0]])
    alloc_min_B = int(min_vol_allocation.at['allocation', stocks[1]])
    alloc_min_C = int(min_vol_allocation.at['allocation', stocks[2]])
    alloc_min_D = int(min_vol_allocation.at['allocation', stocks[3]])
    # Extracting allocation weight for Sharpe Ratio
    global alloc_sr_A
    global alloc_sr_B
    global alloc_sr_C
    global alloc_sr_D

    alloc_sr_A = int(max_sharpe_allocation.at['allocation', stocks[0]])
    alloc_sr_B = int(max_sharpe_allocation.at['allocation', stocks[1]])
    alloc_sr_C = int(max_sharpe_allocation.at['allocation', stocks[2]])
    alloc_sr_D = int(max_sharpe_allocation.at['allocation', stocks[3]])


# Taking in principle amount
# print("\n\n\n")
# print("-" * 80)
# print('Enter the principle amount you would like to invest in USD \n')

# print("-" * 80)


# # Allocation of principle amount based on minumum volatility

# princealloc_min_A = int((alloc_min_A/100) * principle)
# princealloc_min_B = int((alloc_min_B/100) * principle)
# princealloc_min_C = int((alloc_min_C/100) * principle)
# princealloc_min_D = int((alloc_min_D/100) * principle)

# Allocation of principle amount based on Sharpe Ratio


# # Displaying the capital allocation for the Portfolio obtained based on minimum volatility
# print("\n\n\n")
# print("-" * 80)
# print("Minimum Volatility Portfolio Capital Allocation \n")
# print("AAPL", end=' : ')
# print(princealloc_min_A, "USD")
# print("AMZN", end=' : ')
# print(princealloc_min_B, "USD")
# print("GOOGL", end=' : ')
# print(princealloc_min_C, "USD")
# print("FB", end=' : ')
# print(princealloc_min_D, "USD")
# print("\n")
# print("-" * 80)


# # Displaying the capital allocation for the Portfolio obtained based on Sharpe Ratio
# print("\n\n\n")
# print("-" * 80)
# print("Sharpe Ratio Portfolio Capital Allocation \n")
# print("AAPL", end=' : ')
# print(princealloc_sr_A, "USD")
# print("AMZN", end=' : ')
# print(princealloc_sr_B, "USD")
# print("GOOGL", end=' : ')
# print(princealloc_sr_C, "USD")
# print("FB", end=' : ')
# print(princealloc_sr_C, "USD")
# print("\n")
# print("-" * 80)


# # Displaying the capital allocation for the Benchmark Portfolio
# print("\n\n\n")
# print("-" * 80)
# print("Benchmark Portfolio Capital Allocation \n")
# print("AAPL", end=' : ')
# print(principle//4, "USD")
# print("AMZN", end=' : ')
# print(principle//4, "USD")
# print("GOOGL", end=' : ')
# print(principle//4, "USD")
# print("FB", end=' : ')
# print(principle//4, "USD")
# print("\n")
# print("-" * 80)
stocksList = ['A', 'AA', 'AAL', 'AAMC', 'AAN', 'AAOI', 'AAON', 'AAP', 'AAPL', 'AAT', 'AAWW', 'ABAX', 'ABBV', 'ABC', 'ABCB', 'ABCO', 'ABFS', 'ABG', 'ABM', 'ABMD', 'ABT', 'ACAD', 'ACAS', 'ACAT', 'ACC', 'ACCL', 'ACCO', 'ACE', 'ACET', 'ACFN', 'ACGL', 'ACHC', 'ACHN', 'ACI', 'ACIW', 'ACLS', 'ACM', 'ACN', 'ACO', 'ACOR', 'ACRE', 'ACRX', 'ACTG', 'ACW', 'ACXM', 'ADBE', 'ADC', 'ADES', 'ADI', 'ADM', 'ADMS', 'ADNC', 'ADP', 'ADS', 'ADSK', 'ADT', 'ADTN', 'ADUS', 'ADVS', 'AE', 'AEC', 'AEE', 'AEGN', 'AEGR', 'AEIS', 'AEL', 'AEO', 'AEP', 'AEPI', 'AERI', 'AES', 'AET', 'AF', 'AFAM', 'AFFX', 'AFG', 'AFH', 'AFL', 'AFOP', 'AFSI', 'AGCO', 'AGEN', 'AGII', 'AGIO', 'AGM', 'AGN', 'AGNC', 'AGO', 'AGTC', 'AGX', 'AGYS', 'AHC', 'AHH', 'AHL', 'AHP', 'AHS', 'AHT', 'AI', 'AIG', 'AIMC', 'AIN', 'AINV', 'AIQ', 'AIR', 'AIRM', 'AIT', 'AIV', 'AIZ', 'AJG', 'AKAM', 'AKAO', 'AKBA', 'AKR', 'AKRX', 'AKS', 'AL', 'ALB', 'ALCO', 'ALDR', 'ALE', 'ALEX', 'ALG', 'ALGN', 'ALGT', 'ALIM', 'ALJ', 'ALK', 'ALKS', 'ALL', 'ALLE', 'ALNY', 'ALOG', 'ALR', 'ALSN', 'ALTR', 'ALX', 'ALXN', 'AMAG', 'AMAT', 'AMBA', 'AMBC', 'AMBR', 'AMC', 'AMCC', 'AMCX', 'AMD', 'AME', 'AMED', 'AMG', 'AMGN', 'AMKR', 'AMNB', 'AMP', 'AMPE', 'AMRC', 'AMRE', 'AMRI', 'AMRS', 'AMSC', 'AMSF', 'AMSG', 'AMSWA', 'AMT', 'AMTD', 'AMTG', 'AMWD', 'AMZG', 'AMZN', 'AN', 'ANAC', 'ANAD', 'ANAT', 'ANCX', 'ANDE', 'ANDV', 'ANF', 'ANGI', 'ANGO', 'ANH', 'ANIK', 'ANIP', 'ANN', 'ANR', 'ANSS', 'ANTM', 'ANV', 'AOI', 'AOL', 'AON', 'AOS', 'AOSL', 'AP', 'APA', 'APAGF', 'APAM', 'APC', 'APD', 'APEI', 'APH', 'APL', 'APOG', 'APOL', 'APP', 'APTV', 'ARAY', 'ARC', 'ARCB', 'ARCC', 'ARCW', 'ARE', 'AREX', 'ARG', 'ARI', 'ARIA', 'ARII', 'ARNA', 'ARNC', 'ARO', 'AROW', 'ARPI', 'ARQL', 'ARR', 'ARRS', 'ARRY', 'ARSD', 'ARTC', 'ARTNA', 'ARUN', 'ARW', 'ARWR', 'ARX', 'ASBC', 'ASC', 'ASCMA', 'ASEI', 'ASGN', 'ASH', 'ASNA', 'ASPS', 'ASPX', 'ASTE', 'AT', 'ATEC', 'ATEN', 'ATHN', 'ATI', 'ATK', 'ATLO', 'ATMI', 'ATML', 'ATNI', 'ATNM', 'ATNY', 'ATO', 'ATR', 'ATRC', 'ATRI', 'ATRO', 'ATRS', 'ATSG', 'ATU', 'ATVI', 'ATW', 'AUXL', 'AVA', 'AVAV', 'AVB', 'AVD', 'AVEO', 'AVG', 'AVGO', 'AVHI', 'AVID', 'AVIV', 'AVNR', 'AVNW', 'AVP', 'AVT', 'AVX', 'AVY', 'AWAY', 'AWH', 'AWI', 'AWK', 'AWR', 'AXAS', 'AXDX', 'AXE', 'AXL', 'AXLL', 'AXP', 'AXS', 'AYI', 'AYR', 'AZO', 'AZPN', 'AZZ', 'B', 'BA', 'BABY', 'BAC', 'BAGL', 'BAGR', 'BAH', 'BALT', 'BANC', 'BANF', 'BANR', 'BAS', 'BAX', 'BBBY', 'BBCN', 'BBG', 'BBGI', 'BBNK', 'BBOX', 'BBRG', 'BBSI', 'BBT', 'BBW', 'BBX', 'BBY', 'BC', 'BCC', 'BCEI', 'BCO', 'BCOR', 'BCOV', 'BCPC', 'BCR', 'BCRX', 'BDBD', 'BDC', 'BDE', 'BDGE', 'BDN', 'BDSI', 'BDX', 'BEAM', 'BEAT', 'BEAV', 'BEBE', 'BECN', 'BEE', 'BELFB', 'BEN', 'BERY', 'BFAM', 'BFIN', 'BFS', 'BF_B', 'BG', 'BGC', 'BGCP', 'BGFV', 'BGG', 'BGS', 'BH', 'BHB', 'BHE', 'BHF', 'BHGE', 'BHI', 'BHLB', 'BID', 'BIDU', 'BIG', 'BIIB', 'BIO', 'BIOL', 'BIOS', 'BIRT', 'BJRI', 'BK', 'BKCC', 'BKD', 'BKE', 'BKH', 'BKMU', 'BKNG', 'BKS', 'BKU', 'BKW', 'BKYF', 'BLDR', 'BLK', 'BLKB', 'BLL', 'BLMN', 'BLOX', 'BLT', 'BLUE', 'BLX', 'BMI', 'BMR', 'BMRC', 'BMRN', 'BMS', 'BMTC', 'BMY', 'BNCL', 'BNCN', 'BNFT', 'BNNY', 'BOBE', 'BODY', 'BOFI', 'BOH', 'BOKF', 'BOLT', 'BONT', 'BOOM', 'BP', 'BPFH', 'BPI', 'BPOP', 'BPTH', 'BPZ', 'BR', 'BRC', 'BRCD', 'BRCM', 'BRDR', 'BREW', 'BRKL', 'BRKR', 'BRKS', 'BRK_A', 'BRK_B', 'BRLI', 'BRO', 'BRS', 'BRSS', 'BRT', 'BSET', 'BSFT', 'BSRR', 'BSTC', 'BSX', 'BTH', 'BTU', 'BTX', 'BURL', 'BUSE', 'BV', 'BWA', 'BWC', 'BWINB', 'BWLD', 'BWS', 'BXC', 'BXLT', 'BXP', 'BXS', 'BYD', 'BYI', 'BZH', 'C', 'CA', 'CAB', 'CAC', 'CACB', 'CACC', 'CACI', 'CACQ', 'CAG', 'CAH', 'CAKE', 'CALD', 'CALL', 'CALM', 'CALX', 'CAM', 'CAMP', 'CAP', 'CAR', 'CARA', 'CARB', 'CAS', 'CASH', 'CASS', 'CASY', 'CAT', 'CATM', 'CATO', 'CATY', 'CAVM', 'CB', 'CBB', 'CBEY', 'CBF', 'CBG', 'CBI', 'CBK', 'CBL', 'CBM', 'CBOE', 'CBPX', 'CBR', 'CBRE', 'CBRL', 'CBS', 'CBSH', 'CBST', 'CBT', 'CBU', 'CBZ', 'CCBG', 'CCC', 'CCE', 'CCF', 'CCG', 'CCI', 'CCK', 'CCL', 'CCMP', 'CCNE', 'CCO', 'CCOI', 'CCRN', 'CCXI', 'CDE', 'CDI', 'CDNS', 'CDR', 'CE', 'CEB', 'CECE', 'CECO', 'CELG', 'CEMP', 'CENTA', 'CENX', 'CERN', 'CERS', 'CETV', 'CEVA', 'CF', 'CFFI', 'CFFN', 'CFG', 'CFI', 'CFN', 'CFNB', 'CFNL', 'CFR', 'CFX', 'CGI', 'CGNX', 'CHCO', 'CHD', 'CHDN', 'CHDX', 'CHE', 'CHEF', 'CHFC', 'CHFN', 'CHGG', 'CHH', 'CHK', 'CHKP', 'CHMG', 'CHMT', 'CHRW', 'CHS', 'CHSP', 'CHTP', 'CHTR', 'CHUY', 'CI', 'CIA', 'CIDM', 'CIE', 'CIEN', 'CIFC', 'CIM', 'CINF', 'CIR', 'CIT', 'CIX', 'CJES', 'CKEC', 'CKH', 'CKP', 'CL', 'CLC', 'CLCT', 'CLD', 'CLDT', 'CLDX', 'CLF', 'CLFD', 'CLGX', 'CLH', 'CLI', 'CLMS', 'CLNE', 'CLNY', 'CLR', 'CLUB', 'CLVS', 'CLW', 'CLX', 'CMA', 'CMC', 'CMCO', 'CMCSA', 'CMCSK', 'CME', 'CMG', 'CMI', 'CMLS', 'CMN', 'CMO', 'CMP', 'CMRX', 'CMS', 'CMTL', 'CNA', 'CNBC', 'CNBKA', 'CNC', 'CNDO', 'CNK', 'CNL', 'CNMD', 'CNO', 'CNOB', 'CNP', 'CNQR', 'CNS', 'CNSI', 'CNSL', 'CNVR', 'CNW', 'CNX', 'COB', 'COBZ', 'COCO', 'CODE', 'COF', 'COG', 'COH', 'COHR', 'COHU', 'COKE', 'COL', 'COLB', 'COLM', 'CONE', 'CONN', 'COO', 'COP', 'COR', 'CORE', 'CORR', 'CORT', 'COST', 'COTY', 'COUP', 'COV', 'COVS', 'COWN', 'CPA', 'CPB', 'CPE', 'CPF', 'CPGX', 'CPHD', 'CPK', 'CPLA', 'CPN', 'CPRT', 'CPS', 'CPSI', 'CPSS', 'CPST', 'CPT', 'CPWR', 'CQB', 'CR', 'CRAI', 'CRAY', 'CRCM', 'CRD_B', 'CREE', 'CRI', 'CRIS', 'CRK', 'CRL', 'CRM', 'CRMT', 'CROX', 'CRR', 'CRRC', 'CRRS', 'CRS', 'CRUS', 'CRVL', 'CRWN', 'CRY', 'CRZO', 'CSBK', 'CSC', 'CSCD', 'CSCO', 'CSE', 'CSFL', 'CSG', 'CSGP', 'CSGS', 'CSH', 'CSII', 'CSL', 'CSLT', 'CSOD', 'CSRA', 'CSS', 'CST', 'CSU', 'CSV', 'CSWC', 'CSX', 'CTAS', 'CTB', 'CTBI', 'CTCT', 'CTG', 'CTIC', 'CTL', 'CTO', 'CTRE', 'CTRL', 'CTRN', 'CTRX', 'CTS', 'CTSH', 'CTT', 'CTWS', 'CTXS', 'CUB', 'CUBE', 'CUBI', 'CUDA', 'CUI', 'CUNB', 'CUR', 'CUTR', 'CUZ', 'CVA', 'CVBF', 'CVC', 'CVCO', 'CVD', 'CVEO', 'CVG', 'CVGI', 'CVGW', 'CVI', 'CVLT', 'CVO', 'CVS', 'CVT', 'CVX', 'CW', 'CWCO', 'CWEI', 'CWH', 'CWST', 'CWT', 'CXO', 'CXW', 'CY', 'CYBX', 'CYH', 'CYN', 'CYNI', 'CYNO', 'CYS', 'CYT', 'CYTK', 'CYTR', 'CYTX', 'CZNC', 'CZR', 'D', 'DAKT', 'DAL', 'DAN', 'DAR', 'DATA', 'DAVE', 'DBD', 'DCI', 'DCO', 'DCOM', 'DCT', 'DD', 'DDD', 'DDR', 'DDS', 'DE', 'DECK', 'DEI', 'DEL', 'DELL', 'DENN', 'DEPO', 'DEST', 'DF', 'DFRG', 'DFS', 'DFT', 'DFZ', 'DG', 'DGAS', 'DGI', 'DGICA', 'DGII', 'DGX', 'DHI', 'DHIL', 'DHR', 'DHT', 'DHX', 'DIN', 'DIOD', 'DIS', 'DISCA', 'DISCK', 'DISH', 'DJCO', 'DK', 'DKS', 'DLB', 'DLLR', 'DLPH', 'DLR', 'DLTR', 'DLX', 'DMD', 'DMND', 'DMRC', 'DNB', 'DNDN', 'DNKN', 'DNR', 'DO', 'DOC', 'DOOR', 'DORM', 'DOV', 'DOW', 'DOX', 'DPS', 'DPZ', 'DRC', 'DRE', 'DRH', 'DRI', 'DRII', 'DRIV', 'DRL', 'DRNA', 'DRQ', 'DRTX', 'DSCI', 'DSPG', 'DST', 'DSW', 'DTE', 'DTLK', 'DTSI', 'DTV', 'DUK', 'DV', 'DVA', 'DVAX', 'DVN', 'DVR', 'DW', 'DWA', 'DWDP', 'DWRE', 'DWSN', 'DX', 'DXC', 'DXCM', 'DXLG', 'DXM', 'DXPE', 'DXYN', 'DY', 'DYAX', 'DYN', 'EA', 'EAC', 'EAT', 'EBAY', 'EBF', 'EBIO', 'EBIX', 'EBS', 'EBSB', 'EBTC', 'ECHO', 'ECL', 'ECOL', 'ECOM', 'ECPG', 'ECYT', 'ED', 'EDE', 'EDIG', 'EDMC', 'EDR', 'EE', 'EEFT', 'EFII', 'EFSC', 'EFX', 'EGAN', 'EGBN', 'EGHT', 'EGL', 'EGLT', 'EGN', 'EGOV', 'EGP', 'EGY', 'EHTH', 'EIG', 'EIGI', 'EIX', 'EL', 'ELGX', 'ELLI', 'ELNK', 'ELRC', 'ELS', 'ELX', 'ELY', 'EMC', 'EMCI', 'EME', 'EMN', 'EMR', 'END', 'ENDP', 'ENH', 'ENOC', 'ENPH', 'ENR', 'ENS', 'ENSG', 'ENT', 'ENTA', 'ENTG', 'ENTR', 'ENV', 'ENVE', 'ENZ', 'ENZN', 'EOG', 'EOPN', 'EOX', 'EPAM', 'EPAY', 'EPIQ', 'EPL', 'EPM', 'EPR', 'EPZM', 'EQIX', 'EQR', 'EQT', 'EQU', 'EQY', 'ERA', 'ERIE', 'ERII', 'EROS', 'ES', 'ESBF', 'ESC', 'ESCA', 'ESE', 'ESGR', 'ESI', 'ESIO', 'ESL', 'ESNT', 'ESPR', 'ESRT', 'ESRX', 'ESS', 'ESSA', 'ESV', 'ETFC', 'ETH', 'ETM', 'ETN', 'ETR', 'EV', 'EVC', 'EVDY', 'EVER', 'EVHC', 'EVR', 'EVRY', 'EVTC', 'EW', 'EWBC', 'EXAC', 'EXAM', 'EXAR', 'EXAS', 'EXC', 'EXEL', 'EXH', 'EXL', 'EXLS', 'EXP', 'EXPD', 'EXPE', 'EXPO', 'EXPR', 'EXR', 'EXTR', 'EXXI', 'EZPW', 'F', 'FAF', 'FANG', 'FARM', 'FARO', 'FAST', 'FB', 'FBC', 'FBHS', 'FBIZ', 'FBNC', 'FBNK', 'FBP', 'FBRC', 'FC', 'FCBC', 'FCEL', 'FCE_A', 'FCF', 'FCFS', 'FCH', 'FCN', 'FCNCA', 'FCS', 'FCSC', 'FCX', 'FDEF', 'FDML', 'FDO', 'FDP', 'FDS', 'FDUS', 'FDX', 'FE', 'FEIC', 'FELE', 'FET', 'FF', 'FFBC', 'FFBH', 'FFG', 'FFIC', 'FFIN', 'FFIV', 'FFKT', 'FFNW', 'FGL', 'FHCO', 'FHN', 'FIBK', 'FICO', 'FII', 'FINL', 'FIO', 'FIS', 'FISI', 'FISV', 'FITB', 'FIVE', 'FIVN', 'FIX', 'FIZZ', 'FL', 'FLDM', 'FLIC', 'FLIR', 'FLO', 'FLR', 'FLS', 'FLT', 'FLTX', 'FLWS', 'FLXN', 'FLXS', 'FMBI', 'FMC', 'FMD', 'FMER', 'FMI', 'FN', 'FNB', 'FNF', 'FNFG', 'FNGN', 'FNHC', 'FNLC', 'FNSR', 'FOE', 'FOLD', 'FOR', 'FORM', 'FORR', 'FOSL', 'FOX', 'FOXA', 'FOXF', 'FPO', 'FPRX', 'FR', 'FRAN', 'FRBK', 'FRC', 'FRED', 'FRF', 'FRGI', 'FRM', 'FRME', 'FRNK', 'FRO', 'FRP', 'FRSH', 'FRT', 'FRX', 'FSC', 'FSGI', 'FSL', 'FSLR', 'FSP', 'FSS', 'FST', 'FSTR', 'FSYS', 'FTD', 'FTI', 'FTK', 'FTNT', 'FTR', 'FTV', 'FUBC', 'FUEL', 'FUL', 'FULT', 'FUR', 'FURX', 'FVE', 'FWM', 'FWRD', 'FXCB', 'FXCM', 'FXEN', 'G', 'GABC', 'GAIA', 'GAIN', 'GALE', 'GALT', 'GARS', 'GAS', 'GB', 'GBCI', 'GBDC', 'GBL', 'GBLI', 'GBNK', 'GBX', 'GCA', 'GCAP', 'GCI', 'GCO', 'GD', 'GDOT', 'GDP', 'GE', 'GEF', 'GEO', 'GEOS', 'GERN', 'GES', 'GEVA', 'GFF', 'GFIG', 'GFN', 'GGG', 'GGP', 'GHC', 'GHDX', 'GHL', 'GHM', 'GIFI', 'GIII', 'GILD', 'GIMO', 'GIS', 'GK', 'GLAD', 'GLDD', 'GLF', 'GLNG', 'GLOG', 'GLPW', 'GLRE', 'GLRI', 'GLT', 'GLUU', 'GLW', 'GM', 'GMAN', 'GMCR', 'GME', 'GMED', 'GMO', 'GMT', 'GNC', 'GNCA', 'GNCMA', 'GNE', 'GNMK', 'GNRC', 'GNTX', 'GNW', 'GOGO', 'GOLD', 'GOOD', 'GOOG', 'GOOGL', 'GORO', 'GOV', 'GPC', 'GPI', 'GPK', 'GPN', 'GPOR', 'GPRE', 'GPS', 'GPT', 'GPX', 'GRA', 'GRC', 'GRIF', 'GRMN', 'GRPN', 'GRT', 'GRUB', 'GS', 'GSAT', 'GSBC', 'GSBD', 'GSIG', 'GSIT', 'GSM', 'GSOL', 'GST', 'GSVC', 'GT', 'GTAT', 'GTI', 'GTIV', 'GTLS', 'GTN', 'GTS', 'GTT', 'GTXI', 'GTY', 'GUID', 'GVA', 'GWR', 'GWRE', 'GWW', 'GXP', 'GY', 'H', 'HA', 'HAE', 'HAFC', 'HAIN', 'HAL', 'HALL', 'HALO', 'HAR', 'HAS', 'HASI', 'HAWK', 'HAYN', 'HBAN', 'HBCP', 'HBHC', 'HBI', 'HBIO', 'HBNC', 'HCA', 'HCBK', 'HCC', 'HCCI', 'HCI', 'HCKT', 'HCN', 'HCOM', 'HCP', 'HCSG', 'HCT', 'HD', 'HDNG', 'HE', 'HEAR', 'HEES', 'HEI', 'HELE', 'HELI', 'HEOP', 'HERO', 'HES', 'HF', 'HFC', 'HFWA', 'HGG', 'HGR', 'HHC', 'HHS', 'HI', 'HIBB', 'HIFS', 'HIG', 'HII', 'HIIQ', 'HIL', 'HILL', 'HITK', 'HITT', 'HIVE', 'HIW', 'HK', 'HL', 'HLF', 'HLIT', 'HLS', 'HLSS', 'HLT', 'HLX', 'HME', 'HMHC', 'HMN', 'HMPR', 'HMST', 'HMSY', 'HMTV', 'HNH', 'HNI', 'HNR', 'HNRG', 'HNT', 'HOFT', 'HOG', 'HOLX', 'HOMB', 'HOME', 'HON', 'HOS', 'HOT', 'HOV', 'HP', 'HPE', 'HPP', 'HPQ', 'HPT', 'HPTX', 'HPY', 'HR', 'HRB', 'HRC', 'HRG', 'HRL', 'HRS', 'HRTG', 'HRTX', 'HRZN', 'HSC', 'HSH', 'HSIC', 'HSII', 'HSNI', 'HSP', 'HST', 'HSTM', 'HSY', 'HT', 'HTA', 'HTBI', 'HTBK', 'HTCH', 'HTCO', 'HTGC', 'HTH', 'HTLD', 'HTLF', 'HTS', 'HTWR', 'HTZ', 'HUBG', 'HUB_B', 'HUM', 'HUN', 'HURC', 'HURN', 'HVB', 'HVT', 'HW', 'HWAY', 'HWCC', 'HWKN', 'HXL', 'HY', 'HZNP', 'HZO', 'I', 'IACI', 'IART', 'IBCA', 'IBCP', 'IBKC', 'IBKR', 'IBM', 'IBOC', 'IBP', 'IBTX', 'ICE', 'ICEL', 'ICFI', 'ICGE', 'ICON', 'ICPT', 'ICUI', 'IDA', 'IDCC', 'IDIX', 'IDRA', 'IDT', 'IDTI', 'IDXX', 'IEX', 'IFF', 'IFT', 'IG', 'IGT', 'IGTE', 'IHC', 'IHS', 'III', 'IIIN', 'IILG', 'IIVI', 'IL', 'ILMN', 'IM', 'IMGN', 'IMI', 'IMKTA', 'IMMR', 'IMMU', 'IMN', 'IMPV', 'INAP', 'INCY', 'INDB', 'INFA', 'INFI', 'INFN', 'INFO', 'INGN', 'INGR', 'ININ', 'INN', 'INO', 'INSM', 'INSY', 'INT', 'INTC', 'INTL', 'INTU', 'INTX', 'INVN', 'INWK', 'IO', 'IOSP', 'IP', 'IPAR', 'IPCC', 'IPCM', 'IPG', 'IPGP', 'IPHI', 'IPHS', 'IPI', 'IPXL', 'IQNT', 'IQV', 'IR', 'IRBT', 'IRC', 'IRDM', 'IRET', 'IRF', 'IRG', 'IRM', 'IRWD', 'ISBC', 'ISCA', 'ISH', 'ISIL', 'ISIS', 'ISLE', 'ISRG', 'ISRL', 'ISSC', 'ISSI', 'IT', 'ITC', 'ITCI', 'ITG', 'ITIC', 'ITMN', 'ITRI', 'ITT', 'ITW', 'IVAC', 'IVC', 'IVR', 'IVZ', 'IXYS', 'JACK', 'JAH', 'JAKK', 'JBHT', 'JBL', 'JBLU', 'JBSS', 'JBT', 'JCI', 'JCOM', 'JCP', 'JDSU', 'JEC', 'JGW', 'JIVE', 'JJSF', 'JKHY', 'JLL', 'JMBA', 'JNJ', 'JNPR', 'JNS', 'JNY', 'JOE', 'JONE', 'JOUT', 'JOY', 'JPM', 'JRN', 'JWN', 'JW_A', 'K', 'KAI', 'KALU', 'KAMN', 'KAR', 'KBALB', 'KBH', 'KBR', 'KCG', 'KCLI', 'KEG', 'KELYA', 'KEM', 'KERX', 'KEX', 'KEY', 'KEYW', 'KFRC',
              'KFX', 'KFY', 'KHC', 'KIM', 'KIN', 'KIRK', 'KKD', 'KLAC', 'KMB', 'KMG', 'KMI', 'KMPR', 'KMT', 'KMX', 'KND', 'KNL', 'KNX', 'KO', 'KODK', 'KOG', 'KOP', 'KOPN', 'KORS', 'KOS', 'KPTI', 'KR', 'KRA', 'KRC', 'KRFT', 'KRG', 'KRNY', 'KRO', 'KS', 'KSS', 'KSU', 'KTOS', 'KTWO', 'KVHI', 'KW', 'KWK', 'KWR', 'KYTH', 'L', 'LABL', 'LAD', 'LADR', 'LAMR', 'LANC', 'LAYN', 'LAZ', 'LB', 'LBAI', 'LBMH', 'LBTYA', 'LBY', 'LCI', 'LCNB', 'LCUT', 'LDL', 'LDOS', 'LDR', 'LDRH', 'LE', 'LEA', 'LEAF', 'LECO', 'LEE', 'LEG', 'LEN', 'LF', 'LFUS', 'LFVN', 'LG', 'LGF', 'LGIH', 'LGND', 'LH', 'LHCG', 'LHO', 'LIFE', 'LII', 'LINC', 'LINTA', 'LION', 'LIOX', 'LKFN', 'LKQ', 'LL', 'LLEN', 'LLL', 'LLNW', 'LLTC', 'LLY', 'LM', 'LMCA', 'LMIA', 'LMNR', 'LMNX', 'LMOS', 'LMT', 'LNC', 'LNCE', 'LNDC', 'LNG', 'LNKD', 'LNN', 'LNT', 'LO', 'LOCK', 'LOGM', 'LOPE', 'LORL', 'LOV', 'LOW', 'LPG', 'LPI', 'LPLA', 'LPNT', 'LPSN', 'LPX', 'LQ', 'LQDT', 'LRCX', 'LRN', 'LSCC', 'LSI', 'LSTR', 'LTC', 'LTM', 'LTS', 'LTXC', 'LUB', 'LUK', 'LUV', 'LVLT', 'LVNTA', 'LVS', 'LWAY', 'LXFT', 'LXK', 'LXP', 'LXRX', 'LXU', 'LYB', 'LYTS', 'LYV', 'LZB', 'M', 'MA', 'MAA', 'MAC', 'MACK', 'MAIN', 'MAN', 'MANH', 'MANT', 'MAR', 'MAS', 'MASI', 'MAT', 'MATW', 'MATX', 'MBFI', 'MBI', 'MBII', 'MBRG', 'MBUU', 'MBVT', 'MBWM', 'MC', 'MCBC', 'MCC', 'MCD', 'MCF', 'MCGC', 'MCHP', 'MCHX', 'MCK', 'MCO', 'MCP', 'MCRI', 'MCRL', 'MCRS', 'MCS', 'MCY', 'MD', 'MDAS', 'MDC', 'MDCA', 'MDCI', 'MDCO', 'MDLZ', 'MDP', 'MDR', 'MDRX', 'MDSO', 'MDT', 'MDU', 'MDVN', 'MDW', 'MDXG', 'MEAS', 'MED', 'MEG', 'MEI', 'MEIP', 'MENT', 'MET', 'METR', 'MFA', 'MFLX', 'MFRM', 'MG', 'MGAM', 'MGEE', 'MGI', 'MGLN', 'MGM', 'MGNX', 'MGRC', 'MHFI', 'MHGC', 'MHK', 'MHLD', 'MHO', 'MHR', 'MIDD', 'MIG', 'MIL', 'MILL', 'MIND', 'MINI', 'MITK', 'MITT', 'MJN', 'MKC', 'MKL', 'MKSI', 'MKTO', 'MKTX', 'MLAB', 'MLHR', 'MLI', 'MLM', 'MLNK', 'MLR', 'MM', 'MMC', 'MMI', 'MMM', 'MMS', 'MMSI', 'MN', 'MNI', 'MNK', 'MNKD', 'MNR', 'MNRO', 'MNST', 'MNTA', 'MNTX', 'MO', 'MOD', 'MODN', 'MOFG', 'MOG_A', 'MOH', 'MON', 'MORN', 'MOS', 'MOSY', 'MOV', 'MOVE', 'MPAA', 'MPC', 'MPO', 'MPW', 'MPWR', 'MPX', 'MRC', 'MRCY', 'MRGE', 'MRH', 'MRIN', 'MRK', 'MRLN', 'MRO', 'MRTN', 'MRTX', 'MRVL', 'MS', 'MSA', 'MSCC', 'MSCI', 'MSEX', 'MSFG', 'MSFT', 'MSG', 'MSI', 'MSL', 'MSM', 'MSO', 'MSTR', 'MTB', 'MTD', 'MTDR', 'MTG', 'MTGE', 'MTH', 'MTN', 'MTOR', 'MTRN', 'MTRX', 'MTSC', 'MTSI', 'MTW', 'MTX', 'MTZ', 'MU', 'MUR', 'MUSA', 'MVC', 'MVNR', 'MW', 'MWA', 'MWIV', 'MWV', 'MWW', 'MXIM', 'MXL', 'MXWL', 'MYCC', 'MYE', 'MYGN', 'MYL', 'MYRG', 'N', 'NADL', 'NANO', 'NASB', 'NAT', 'NATH', 'NATI', 'NATL', 'NATR', 'NAV', 'NAVB', 'NAVG', 'NAVI', 'NBBC', 'NBCB', 'NBHC', 'NBIX', 'NBL', 'NBR', 'NBS', 'NBTB', 'NC', 'NCFT', 'NCI', 'NCLH', 'NCMI', 'NCR', 'NCS', 'NDAQ', 'NDLS', 'NDSN', 'NE', 'NEE', 'NEM', 'NEO', 'NEOG', 'NEON', 'NES', 'NETE', 'NEU', 'NEWM', 'NEWP', 'NEWS', 'NFBK', 'NFG', 'NFLX', 'NFX', 'NGHC', 'NGPC', 'NGS', 'NGVC', 'NHC', 'NHI', 'NI', 'NICK', 'NIHD', 'NILE', 'NJR', 'NKE', 'NKSH', 'NKTR', 'NL', 'NLNK', 'NLS', 'NLSN', 'NLY', 'NM', 'NMBL', 'NMFC', 'NMIH', 'NMRX', 'NNA', 'NNBR', 'NNI', 'NNVC', 'NOC', 'NOG', 'NOR', 'NOV', 'NOW', 'NP', 'NPBC', 'NPK', 'NPO', 'NPSP', 'NPTN', 'NR', 'NRCIA', 'NRG', 'NRIM', 'NRZ', 'NSC', 'NSIT', 'NSM', 'NSP', 'NSPH', 'NSR', 'NSTG', 'NTAP', 'NTCT', 'NTGR', 'NTK', 'NTLS', 'NTRI', 'NTRS', 'NU', 'NUAN', 'NUE', 'NUS', 'NUTR', 'NUVA', 'NVAX', 'NVDA', 'NVEC', 'NVR', 'NWBI', 'NWBO', 'NWE', 'NWHM', 'NWL', 'NWLI', 'NWN', 'NWPX', 'NWS', 'NWSA', 'NWY', 'NX', 'NXST', 'NXTM', 'NYCB', 'NYLD', 'NYMT', 'NYNY', 'NYRT', 'NYT', 'O', 'OABC', 'OAS', 'OB', 'OC', 'OCFC', 'OCLR', 'OCN', 'OCR', 'ODC', 'ODFL', 'ODP', 'OEH', 'OFC', 'OFG', 'OFIX', 'OFLX', 'OGE', 'OGS', 'OGXI', 'OHI', 'OHRP', 'OI', 'OII', 'OIS', 'OKE', 'OKSB', 'OLBK', 'OLED', 'OLN', 'OLP', 'OMC', 'OMCL', 'OME', 'OMED', 'OMER', 'OMEX', 'OMG', 'OMI', 'OMN', 'ONB', 'ONE', 'ONNN', 'ONTY', 'ONVO', 'OPB', 'OPEN', 'OPHT', 'OPK', 'OPLK', 'OPWR', 'OPY', 'ORA', 'ORB', 'ORBC', 'ORCL', 'OREX', 'ORI', 'ORIT', 'ORLY', 'ORM', 'ORN', 'OSIR', 'OSIS', 'OSK', 'OSTK', 'OSUR', 'OTTR', 'OUTR', 'OVAS', 'OVTI', 'OWW', 'OXFD', 'OXM', 'OXY', 'OZRK', 'P', 'PACB', 'PACR', 'PACW', 'PAG', 'PAHC', 'PANW', 'PATK', 'PATR', 'PAY', 'PAYC', 'PAYX', 'PB', 'PBCT', 'PBF', 'PBH', 'PBI', 'PBPB', 'PBY', 'PBYI', 'PCAR', 'PCBK', 'PCCC', 'PCG', 'PCH', 'PCL', 'PCLN', 'PCO', 'PCP', 'PCRX', 'PCTI', 'PCTY', 'PCYC', 'PCYG', 'PCYO', 'PDCE', 'PDCO', 'PDFS', 'PDLI', 'PDM', 'PE', 'PEB', 'PEBO', 'PEG', 'PEGA', 'PEGI', 'PEI', 'PEIX', 'PENN', 'PENX', 'PEP', 'PERI', 'PERY', 'PES', 'PETM', 'PETS', 'PETX', 'PF', 'PFBC', 'PFE', 'PFG', 'PFIE', 'PFIS', 'PFLT', 'PFMT', 'PFPT', 'PFS', 'PFSI', 'PG', 'PGC', 'PGEM', 'PGI', 'PGNX', 'PGR', 'PGTI', 'PH', 'PHH', 'PHIIK', 'PHM', 'PHMD', 'PHX', 'PICO', 'PII', 'PIKE', 'PIR', 'PJC', 'PKD', 'PKE', 'PKG', 'PKI', 'PKOH', 'PKT', 'PKY', 'PL', 'PLAB', 'PLCE', 'PLCM', 'PLD', 'PLKI', 'PLL', 'PLMT', 'PLOW', 'PLPC', 'PLPM', 'PLT', 'PLUG', 'PLUS', 'PLXS', 'PLXT', 'PM', 'PMC', 'PMCS', 'PMFG', 'PMT', 'PNC', 'PNFP', 'PNK', 'PNM', 'PNNT', 'PNR', 'PNRA', 'PNW', 'PNX', 'PNY', 'PODD', 'POL', 'POM', 'POOL', 'POR', 'POST', 'POWI', 'POWL', 'POWR', 'POZN', 'PPBI', 'PPC', 'PPG', 'PPHM', 'PPL', 'PPO', 'PPS', 'PQ', 'PRA', 'PRAA', 'PRE', 'PRFT', 'PRGO', 'PRGS', 'PRGX', 'PRI', 'PRIM', 'PRK', 'PRKR', 'PRLB', 'PRO', 'PROV', 'PRSC', 'PRTA', 'PRU', 'PRXL', 'PSA', 'PSB', 'PSEC', 'PSEM', 'PSIX', 'PSMI', 'PSMT', 'PSTB', 'PSUN', 'PSX', 'PTCT', 'PTEN', 'PTGI', 'PTIE', 'PTLA', 'PTP', 'PTRY', 'PTSI', 'PTX', 'PVA', 'PVH', 'PVTB', 'PWOD', 'PWR', 'PX', 'PXD', 'PYPL', 'PZG', 'PZN', 'PZZA', 'Q', 'QADA', 'QCOM', 'QCOR', 'QDEL', 'QEP', 'QGEN', 'QLGC', 'QLIK', 'QLTY', 'QLYS', 'QNST', 'QRHC', 'QRVO', 'QSII', 'QTM', 'QTS', 'QTWO', 'QUAD', 'QUIK', 'R', 'RAD', 'RAI', 'RAIL', 'RALY', 'RARE', 'RAS', 'RATE', 'RAVN', 'RAX', 'RBC', 'RBCAA', 'RBCN', 'RCAP', 'RCII', 'RCKB', 'RCL', 'RCPT', 'RDC', 'RDEN', 'RDI', 'RDN', 'RDNT', 'RE', 'RECN', 'REG', 'REGI', 'REGN', 'REI', 'REIS', 'RELL', 'REMY', 'REN', 'RENT', 'RES', 'RESI', 'REV', 'REX', 'REXI', 'REXR', 'REXX', 'RF', 'RFMD', 'RFP', 'RGA', 'RGC', 'RGDO', 'RGEN', 'RGLD', 'RGLS', 'RGR', 'RGS', 'RH', 'RHI', 'RHP', 'RHT', 'RIG', 'RIGL', 'RJET', 'RJF', 'RKT', 'RKUS', 'RL', 'RLD', 'RLGY', 'RLI', 'RLJ', 'RLOC', 'RLYP', 'RM', 'RMAX', 'RMBS', 'RMD', 'RMTI', 'RNDY', 'RNET', 'RNG', 'RNR', 'RNST', 'RNWK', 'ROC', 'ROCK', 'ROG', 'ROIAK', 'ROIC', 'ROK', 'ROL', 'ROLL', 'ROP', 'ROSE', 'ROST', 'ROVI', 'RP', 'RPAI', 'RPM', 'RPRX', 'RPT', 'RPTP', 'RPXC', 'RRC', 'RRD', 'RRGB', 'RRTS', 'RS', 'RSE', 'RSG', 'RSH', 'RSO', 'RSPP', 'RST', 'RSTI', 'RSYS', 'RT', 'RTEC', 'RTI', 'RTIX', 'RTK', 'RTN', 'RTRX', 'RUBI', 'RUSHA', 'RUTH', 'RVBD', 'RVLT', 'RVNC', 'RWT', 'RXN', 'RYL', 'RYN', 'S', 'SAAS', 'SAFM', 'SAFT', 'SAH', 'SAIA', 'SAIC', 'SALE', 'SALM', 'SALT', 'SAM', 'SAMG', 'SANM', 'SAPE', 'SASR', 'SATS', 'SAVE', 'SB', 'SBAC', 'SBCF', 'SBGI', 'SBH', 'SBNY', 'SBRA', 'SBSI', 'SBUX', 'SBY', 'SCAI', 'SCBT', 'SCCO', 'SCG', 'SCHL', 'SCHN', 'SCHW', 'SCI', 'SCL', 'SCLN', 'SCM', 'SCMP', 'SCOR', 'SCS', 'SCSC', 'SCSS', 'SCTY', 'SCVL', 'SD', 'SDRL', 'SE', 'SEAC', 'SEAS', 'SEB', 'SEE', 'SEIC', 'SEM', 'SEMG', 'SENEA', 'SF', 'SFBS', 'SFE', 'SFG', 'SFL', 'SFLY', 'SFNC', 'SFXE', 'SFY', 'SGA', 'SGBK', 'SGEN', 'SGI', 'SGK', 'SGM', 'SGMO', 'SGMS', 'SGNT', 'SGY', 'SGYP', 'SHEN', 'SHLD', 'SHLM', 'SHLO', 'SHO', 'SHOO', 'SHOR', 'SHOS', 'SHW', 'SIAL', 'SIF', 'SIG', 'SIGA', 'SIGI', 'SIGM', 'SIMG', 'SIR', 'SIRI', 'SIRO', 'SIVB', 'SIX', 'SJI', 'SJM', 'SJW', 'SKH', 'SKT', 'SKUL', 'SKX', 'SKYW', 'SLAB', 'SLB', 'SLCA', 'SLG', 'SLGN', 'SLH', 'SLM', 'SLRC', 'SLXP', 'SM', 'SMA', 'SMCI', 'SMG', 'SMP', 'SMRT', 'SMTC', 'SN', 'SNA', 'SNAK', 'SNBC', 'SNCR', 'SNDK', 'SNH', 'SNHY', 'SNI', 'SNMX', 'SNOW', 'SNPS', 'SNSS', 'SNTA', 'SNV', 'SNX', 'SO', 'SON', 'SONC', 'SONS', 'SP', 'SPA', 'SPAR', 'SPB', 'SPDC', 'SPF', 'SPG', 'SPGI', 'SPLK', 'SPLS', 'SPN', 'SPNC', 'SPNS', 'SPPI', 'SPR', 'SPRT', 'SPSC', 'SPTN', 'SPW', 'SPWH', 'SPWR', 'SQBG', 'SQBK', 'SQI', 'SQNM', 'SRCE', 'SRCL', 'SRDX', 'SRE', 'SREV', 'SRI', 'SRPT', 'SSD', 'SSI', 'SSNC', 'SSNI', 'SSP', 'SSS', 'SSTK', 'SSYS', 'STAA', 'STAG', 'STAR', 'STBA', 'STBZ', 'STC', 'STCK', 'STE', 'STFC', 'STI', 'STJ', 'STL', 'STLD', 'STML', 'STMP', 'STNG', 'STNR', 'STR', 'STRA', 'STRL', 'STRT', 'STRZA', 'STSA', 'STSI', 'STT', 'STWD', 'STX', 'STZ', 'SUBK', 'SUI', 'SUN', 'SUNE', 'SUNS', 'SUP', 'SUPN', 'SUPX', 'SUSQ', 'SUSS', 'SVU', 'SVVC', 'SWAY', 'SWC', 'SWFT', 'SWHC', 'SWI', 'SWK', 'SWKS', 'SWM', 'SWN', 'SWS', 'SWSH', 'SWX', 'SWY', 'SXC', 'SXI', 'SXT', 'SYA', 'SYBT', 'SYF', 'SYK', 'SYKE', 'SYMC', 'SYNA', 'SYNT', 'SYRG', 'SYUT', 'SYX', 'SYY', 'SZMK', 'SZYM', 'T', 'TAHO', 'TAL', 'TAM', 'TAP', 'TASR', 'TAST', 'TAT', 'TAX', 'TAXI', 'TAYC', 'TBBK', 'TBI', 'TBNK', 'TBPH', 'TCAP', 'TCB', 'TCBI', 'TCBK', 'TCO', 'TCPC', 'TCRD', 'TCS', 'TDC', 'TDG', 'TDS', 'TDW', 'TDY', 'TE', 'TEAR', 'TECD', 'TECH', 'TECUA', 'TEG', 'TEL', 'TEN', 'TER', 'TESO', 'TESS', 'TEX', 'TFM', 'TFSL', 'TFX', 'TG', 'TGE', 'TGH', 'TGI', 'TGNA', 'TGT', 'TGTX', 'THC', 'THFF', 'THG', 'THLD', 'THO', 'THOR', 'THR', 'THRM', 'THRX', 'THS', 'TIBX', 'TICC', 'TIF', 'TILE', 'TIME', 'TIPT', 'TIS', 'TISI', 'TITN', 'TIVO', 'TJX', 'TK', 'TKR', 'TLMR', 'TLYS', 'TMH', 'TMHC', 'TMK', 'TMO', 'TMP', 'TMUS', 'TNAV', 'TNC', 'TNDM', 'TNET', 'TNGO', 'TNK', 'TOL', 'TOWN', 'TOWR', 'TPC', 'TPH', 'TPLM', 'TPR', 'TPRE', 'TPX', 'TQNT', 'TR', 'TRAK', 'TRC', 'TREC', 'TREE', 'TREX', 'TRGP', 'TRGT', 'TRI', 'TRIP', 'TRIV', 'TRK', 'TRLA', 'TRMB', 'TRMK', 'TRMR', 'TRN', 'TRNO', 'TRNX', 'TROW', 'TROX', 'TRR', 'TRS', 'TRST', 'TRUE', 'TRV', 'TRW', 'TRXC', 'TSC', 'TSCO', 'TSLA', 'TSN', 'TSO', 'TSRA', 'TSRE', 'TSRO', 'TSS', 'TSYS', 'TTC', 'TTEC', 'TTEK', 'TTGT', 'TTI', 'TTMI', 'TTPH', 'TTS', 'TTWO', 'TUES', 'TUMI', 'TUP', 'TW', 'TWC', 'TWER', 'TWGP', 'TWI', 'TWIN', 'TWMC', 'TWO', 'TWOU', 'TWTC', 'TWTR', 'TWX', 'TXI', 'TXMD', 'TXN', 'TXRH', 'TXT', 'TXTR', 'TYC', 'TYL', 'TYPE', 'TZOO', 'UA', 'UAA', 'UACL', 'UAL', 'UAM', 'UA_C', 'UBA', 'UBNK', 'UBNT', 'UBSH', 'UBSI', 'UCBI', 'UCFC', 'UCP', 'UCTT', 'UDR', 'UEC', 'UEIC', 'UFCS', 'UFI', 'UFPI', 'UFPT', 'UFS', 'UGI', 'UHAL', 'UHS', 'UHT', 'UIHC', 'UIL', 'UIS', 'ULTA', 'ULTI', 'ULTR', 'UMBF', 'UMH', 'UMPQ', 'UNF', 'UNFI', 'UNH', 'UNIS', 'UNM', 'UNP', 'UNS', 'UNT', 'UNTD', 'UNXL', 'UPIP', 'UPL', 'UPS', 'URBN', 'URG', 'URI', 'URS', 'USAK', 'USAP', 'USB', 'USCR', 'USG', 'USLM', 'USM', 'USMD', 'USMO', 'USNA', 'USPH', 'USTR', 'UTEK', 'UTHR', 'UTI', 'UTIW', 'UTL', 'UTMD', 'UTX', 'UVE', 'UVSP', 'UVV', 'V', 'VAC', 'VAL', 'VAR', 'VASC', 'VC', 'VCRA', 'VCYT', 'VDSI', 'VECO', 'VFC', 'VG', 'VGR', 'VHC', 'VIAB', 'VIAS', 'VICL', 'VICR', 'VITC', 'VIVO', 'VLCCF', 'VLGEA', 'VLO', 'VLY', 'VMC', 'VMEM', 'VMI', 'VMW', 'VNCE', 'VNDA', 'VNO', 'VNTV', 'VOCS', 'VOD', 'VOLC', 'VOXX', 'VOYA', 'VPFG', 'VPG', 'VPRT', 'VR', 'VRA', 'VRNG', 'VRNS', 'VRNT', 'VRSK', 'VRSN', 'VRTS', 'VRTU', 'VRTX', 'VRX', 'VSAR', 'VSAT', 'VSEC', 'VSH', 'VSI', 'VSTM', 'VTG', 'VTL', 'VTNR', 'VTR', 'VTSS', 'VVC', 'VVI', 'VVTV', 'VVUS', 'VZ', 'WAB', 'WABC', 'WAC', 'WAFD', 'WAG', 'WAGE', 'WAIR', 'WAL', 'WASH', 'WAT', 'WBA', 'WBC', 'WBCO', 'WBMD', 'WBS', 'WCC', 'WCG', 'WCIC', 'WCN', 'WD', 'WDAY', 'WDC', 'WDFC', 'WDR', 'WEC', 'WELL', 'WEN', 'WERN', 'WETF', 'WEX', 'WEYS', 'WFC', 'WFD', 'WFM', 'WG', 'WGL', 'WGO', 'WHF', 'WHG', 'WHR', 'WIBC', 'WIFI', 'WIN', 'WINA', 'WIRE', 'WIX', 'WLB', 'WLH', 'WLK', 'WLL', 'WLP', 'WLT', 'WLTW', 'WM', 'WMAR', 'WMB', 'WMC', 'WMGI', 'WMK', 'WMT', 'WNC', 'WNR', 'WOOF', 'WOR', 'WPP', 'WPX', 'WR', 'WRB', 'WRE', 'WRES', 'WRI', 'WRK', 'WRLD', 'WSBC', 'WSBF', 'WSFS', 'WSM', 'WSO', 'WSR', 'WST', 'WSTC', 'WSTL', 'WTBA', 'WTFC', 'WTI', 'WTM', 'WTR', 'WTS', 'WTSL', 'WTW', 'WU', 'WWAV', 'WWD', 'WWE', 'WWW', 'WWWW', 'WY', 'WYN', 'WYNN', 'X', 'XCO', 'XCRA', 'XEC', 'XEL', 'XL', 'XLNX', 'XLRN', 'XLS', 'XNCR', 'XNPT', 'XOM', 'XOMA', 'XON', 'XONE', 'XOOM', 'XOXO', 'XPO', 'XRAY', 'XRM', 'XRX', 'XXIA', 'XXII', 'XYL', 'Y', 'YDKN', 'YELP', 'YHOO', 'YORW', 'YRCW', 'YUM', 'YUME', 'Z', 'ZAGG', 'ZAZA', 'ZBH', 'ZBRA', 'ZEN', 'ZEP', 'ZEUS', 'ZGNX', 'ZIGO', 'ZINC', 'ZION', 'ZIOP', 'ZIXI', 'ZLC', 'ZLTQ', 'ZMH', 'ZNGA', 'ZOES', 'ZQK', 'ZTS', 'ZUMZ']

table = []
returns = []


@app.route("/",  methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if not request.form['principleAmount']:
            return render_template("index.html", ptype="empty", stocks=stocksList, message="Cannot be empty!")
        else:
            principleAmount = request.form['principleAmount']
            stock_one = request.form.get('stockOne')
            stock_two = request.form.get('stockTwo')
            stock_three = request.form.get('stockThree')
            stock_four = request.form.get('stockFour')

            np.random.seed(777)

            quandl.ApiConfig.api_key = '84qJQFf5dTyzjvyxAAyM'
            stocks = [stock_one, stock_two, stock_three, stock_four]
            data = quandl.get_table('WIKI/PRICES', ticker=stocks,
                                    qopts={'columns': [
                                        'date', 'ticker', 'adj_close']},
                                    date={'gte': '2017-1-1', 'lte': '2019-12-31'}, paginate=True)
            global table
            global returns
            df = data.set_index('date')

            table = df.pivot(columns='ticker')
            table.columns = [col[1] for col in table.columns]

            returns = table.pct_change()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            num_portfolios = 25000
            risk_free_rate = 0.0178

            display_simulated_ef_with_random(
                mean_returns, cov_matrix, num_portfolios, risk_free_rate, stocks)
            principleAmount = int(principleAmount)
            princealloc_sr_A = int((alloc_sr_A/100) * principleAmount)
            princealloc_sr_B = int((alloc_sr_B/100) * principleAmount)
            princealloc_sr_C = int((alloc_sr_C/100) * principleAmount)
            princealloc_sr_D = int((alloc_sr_D/100) * principleAmount)
            return render_template("index.html", stocks=stocksList, ptype="submitted", princealloc_sr_A=princealloc_sr_A, princealloc_sr_B=princealloc_sr_B, princealloc_sr_C=princealloc_sr_C, princealloc_sr_D=princealloc_sr_D)
    else:
        return render_template('index.html', stocks=stocksList)
