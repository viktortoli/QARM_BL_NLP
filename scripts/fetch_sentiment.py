"""
Fetch news from Finnhub and analyze sentiment using FinBERT or Claude
"""

import os
import sys
import time
import re
import json
import requests
from datetime import datetime, timedelta
import psycopg2
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# SENTIMENT ANALYZER CONFIGURATION
# Set USE_CLAUDE_SENTIMENT=True to use Claude Haiku instead of FinBERT
# ============================================================
USE_CLAUDE_SENTIMENT = True
CLAUDE_MODEL = "claude-3-haiku-20240307"
# ============================================================


def log(message):
    """Print timestamped log"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# ============================================================
# CEO/EXECUTIVE TO TICKER MAPPING
# High-profile executives whose names often appear in headlines
# ============================================================
CEO_TO_TICKER = {
    # Tech Giants
    'elon musk': 'TSLA',
    'musk': 'TSLA',
    'mark zuckerberg': 'META',
    'zuckerberg': 'META',
    'jensen huang': 'NVDA',
    'huang': 'NVDA',  # Context-dependent but usually Jensen
    'tim cook': 'AAPL',
    'satya nadella': 'MSFT',
    'nadella': 'MSFT',
    'sundar pichai': 'GOOGL',
    'pichai': 'GOOGL',
    'andy jassy': 'AMZN',
    'jassy': 'AMZN',
    'jeff bezos': 'AMZN',
    'bezos': 'AMZN',
    
    # Finance
    'jamie dimon': 'JPM',
    'dimon': 'JPM',
    'warren buffett': 'BRK.B',
    'buffett': 'BRK.B',
    'david solomon': 'GS',
    'brian moynihan': 'BAC',
    
    # Other Notable
    'lisa su': 'AMD',
    'pat gelsinger': 'INTC',
    'gelsinger': 'INTC',
    'mary barra': 'GM',
    'barra': 'GM',
    'doug mcmillon': 'WMT',
    'arvind krishna': 'IBM',
    'bob iger': 'DIS',
    'iger': 'DIS',
    'reed hastings': 'NFLX',
    'greg abel': 'BRK.B',
    'dara khosrowshahi': 'UBER',
    'brian niccol': 'SBUX',
    'laxman narasimhan': 'SBUX',  # Previous CEO
}

# Company name variations for headline matching (case-insensitive)
# Each ticker maps to a list of lowercase aliases that identify the company
COMPANY_ALIASES = {
    # === A ===
    'A': ['agilent', 'agilent technologies'],
    'AAPL': ['apple', 'iphone', 'ipad', 'mac', 'macbook', 'airpods', 'apple watch'],
    'ABBV': ['abbvie', 'humira', 'rinvoq', 'skyrizi'],
    'ABNB': ['airbnb', 'air bnb'],
    'ABT': ['abbott', 'abbott laboratories', 'freestyle libre'],
    'ACGL': ['arch capital', 'arch capital group'],
    'ACN': ['accenture'],
    'ADBE': ['adobe', 'photoshop', 'acrobat', 'creative cloud', 'illustrator'],
    'ADI': ['analog devices'],
    'ADM': ['archer daniels', 'archer-daniels-midland', 'adm'],
    'ADP': ['adp', 'automatic data processing'],
    'ADSK': ['autodesk', 'autocad', 'revit'],
    'AEE': ['ameren', 'ameren corporation'],
    'AEP': ['american electric power', 'aep'],
    'AES': ['aes corporation', 'aes corp'],
    'AFL': ['aflac'],
    'AIG': ['aig', 'american international group'],
    'AIZ': ['assurant'],
    'AJG': ['arthur j gallagher', 'gallagher'],
    'AKAM': ['akamai', 'akamai technologies'],
    'ALB': ['albemarle'],
    'ALGN': ['align technology', 'invisalign'],
    'ALL': ['allstate'],
    'ALLE': ['allegion'],
    'AMAT': ['applied materials'],
    'AMCR': ['amcor'],
    'AMD': ['amd', 'advanced micro devices', 'radeon', 'ryzen', 'epyc'],
    'AME': ['ametek'],
    'AMGN': ['amgen'],
    'AMP': ['ameriprise', 'ameriprise financial'],
    'AMT': ['american tower'],
    'AMZN': ['amazon', 'aws', 'prime', 'whole foods', 'alexa'],
    'ANET': ['arista', 'arista networks'],
    'AON': ['aon', 'aon plc'],
    'AOS': ['a.o. smith', 'ao smith', 'a o smith'],
    'APA': ['apa corporation', 'apache'],
    'APD': ['air products', 'air products and chemicals'],
    'APH': ['amphenol'],
    'APO': ['apollo', 'apollo global', 'apollo global management'],
    'APP': ['applovin'],
    'APTV': ['aptiv'],
    'ARE': ['alexandria real estate', 'alexandria'],
    'ATO': ['atmos energy'],
    'AVB': ['avalonbay', 'avalonbay communities'],
    'AVGO': ['broadcom'],
    'AVY': ['avery dennison', 'avery'],
    'AWK': ['american water', 'american water works'],
    'AXON': ['axon', 'axon enterprise', 'taser'],
    'AXP': ['american express', 'amex'],
    'AZO': ['autozone', 'auto zone'],
    
    # === B ===
    'BA': ['boeing'],
    'BAC': ['bank of america', 'bofa', 'merrill lynch'],
    'BALL': ['ball corporation', 'ball corp'],
    'BAX': ['baxter', 'baxter international'],
    'BBY': ['best buy', 'bestbuy'],
    'BDX': ['becton dickinson', 'becton, dickinson', 'bd'],
    'BEN': ['franklin templeton', 'franklin resources'],
    'BF-B': ['brown-forman', 'brown forman', 'jack daniels'],
    'BG': ['bunge', 'bunge global'],
    'BIIB': ['biogen'],
    'BK': ['bny mellon', 'bank of new york', 'bank of new york mellon'],
    'BKNG': ['booking', 'booking.com', 'priceline', 'kayak'],
    'BKR': ['baker hughes'],
    'BLDR': ['builders firstsource'],
    'BLK': ['blackrock', 'black rock'],
    'BMY': ['bristol-myers', 'bristol myers', 'bristol-myers squibb'],
    'BR': ['broadridge', 'broadridge financial'],
    'BRK-A': ['berkshire', 'berkshire hathaway'],
    'BRO': ['brown & brown'],
    'BSX': ['boston scientific'],
    'BX': ['blackstone'],
    'BXP': ['boston properties'],
    
    # === C ===
    'C': ['citigroup', 'citi', 'citibank'],
    'CAG': ['conagra', 'conagra brands'],
    'CAH': ['cardinal health'],
    'CARR': ['carrier', 'carrier global'],
    'CAT': ['caterpillar'],
    'CB': ['chubb'],
    'CBOE': ['cboe', 'cboe global', 'chicago board options'],
    'CBRE': ['cbre', 'cbre group'],
    'CCI': ['crown castle'],
    'CCL': ['carnival', 'carnival cruise', 'carnival corporation'],
    'CDNS': ['cadence', 'cadence design', 'cadence design systems'],
    'CDW': ['cdw', 'cdw corporation'],
    'CEG': ['constellation energy'],
    'CF': ['cf industries'],
    'CFG': ['citizens financial', 'citizens bank'],
    'CHD': ['church & dwight', 'church and dwight', 'arm & hammer'],
    'CHRW': ['c.h. robinson', 'ch robinson'],
    'CHTR': ['charter', 'charter communications', 'spectrum'],
    'CI': ['cigna', 'the cigna group'],
    'CINF': ['cincinnati financial'],
    'CL': ['colgate', 'colgate-palmolive'],
    'CLX': ['clorox'],
    'CMCSA': ['comcast', 'xfinity', 'nbcuniversal'],
    'CME': ['cme group', 'chicago mercantile'],
    'CMG': ['chipotle', 'chipotle mexican grill'],
    'CMI': ['cummins'],
    'CMS': ['cms energy'],
    'CNC': ['centene'],
    'CNP': ['centerpoint', 'centerpoint energy'],
    'COF': ['capital one'],
    'COIN': ['coinbase'],
    'COO': ['cooper', 'cooper companies'],
    'COP': ['conocophillips', 'conoco'],
    'COR': ['cencora'],
    'COST': ['costco'],
    'CPAY': ['corpay'],
    'CPB': ['campbell', 'campbell soup', "campbell's"],
    'CPRT': ['copart'],
    'CPT': ['camden property', 'camden property trust'],
    'CRL': ['charles river', 'charles river laboratories'],
    'CRM': ['salesforce'],
    'CRWD': ['crowdstrike', 'crowd strike'],
    'CSCO': ['cisco', 'cisco systems'],
    'CSGP': ['costar', 'costar group'],
    'CSX': ['csx', 'csx corporation'],
    'CTAS': ['cintas'],
    'CTRA': ['coterra', 'coterra energy'],
    'CTSH': ['cognizant', 'cognizant technology'],
    'CTVA': ['corteva', 'corteva agriscience'],
    'CVS': ['cvs', 'cvs health', 'cvs pharmacy'],
    'CVX': ['chevron'],
    
    # === D ===
    'D': ['dominion', 'dominion energy'],
    'DAL': ['delta', 'delta air lines', 'delta airlines'],
    'DASH': ['doordash', 'door dash'],
    'DAY': ['dayforce'],
    'DD': ['dupont', 'du pont'],
    'DDOG': ['datadog', 'data dog'],
    'DE': ['john deere', 'deere', 'deere & company'],
    'DECK': ['deckers', 'deckers outdoor', 'ugg', 'hoka'],
    'DELL': ['dell', 'dell technologies'],
    'DG': ['dollar general'],
    'DGX': ['quest diagnostics', 'quest'],
    'DHI': ['d.r. horton', 'dr horton'],
    'DHR': ['danaher'],
    'DIS': ['disney', 'walt disney', 'disney+', 'espn', 'marvel', 'pixar', 'hulu'],
    'DLR': ['digital realty'],
    'DLTR': ['dollar tree'],
    'DOC': ['healthpeak', 'healthpeak properties'],
    'DOV': ['dover', 'dover corporation'],
    'DOW': ['dow', 'dow inc', 'dow chemical'],
    'DPZ': ['dominos', "domino's", 'domino pizza'],
    'DRI': ['darden', 'darden restaurants', 'olive garden'],
    'DTE': ['dte energy'],
    'DUK': ['duke energy'],
    'DVA': ['davita'],
    'DVN': ['devon', 'devon energy'],
    'DXCM': ['dexcom'],
    
    # === E ===
    'EA': ['electronic arts', 'ea sports', 'ea games'],
    'EBAY': ['ebay'],
    'ECL': ['ecolab'],
    'ED': ['consolidated edison', 'con edison', 'coned'],
    'EFX': ['equifax'],
    'EG': ['everest group'],
    'EIX': ['edison international', 'edison'],
    'EL': ['estee lauder', 'estée lauder'],
    'ELV': ['elevance', 'elevance health', 'anthem'],
    'EME': ['emcor', 'emcor group'],
    'EMN': ['eastman chemical', 'eastman'],
    'EMR': ['emerson', 'emerson electric'],
    'EOG': ['eog resources', 'eog'],
    'EPAM': ['epam', 'epam systems'],
    'EQIX': ['equinix'],
    'EQR': ['equity residential'],
    'EQT': ['eqt', 'eqt corporation'],
    'ERIE': ['erie insurance', 'erie indemnity'],
    'ES': ['eversource', 'eversource energy'],
    'ESS': ['essex property', 'essex property trust'],
    'ETN': ['eaton', 'eaton corporation'],
    'ETR': ['entergy'],
    'EVRG': ['evergy'],
    'EW': ['edwards lifesciences', 'edwards'],
    'EXC': ['exelon'],
    'EXE': ['expand energy'],
    'EXPD': ['expeditors', 'expeditors international'],
    'EXPE': ['expedia', 'expedia group', 'vrbo', 'hotels.com'],
    'EXR': ['extra space storage', 'extra space'],
    
    # === F ===
    'F': ['ford', 'ford motor', 'f-150', 'mustang', 'bronco'],
    'FANG': ['diamondback', 'diamondback energy'],
    'FAST': ['fastenal'],
    'FCX': ['freeport-mcmoran', 'freeport mcmoran', 'freeport'],
    'FDS': ['factset', 'factset research'],
    'FDX': ['fedex', 'federal express'],
    'FE': ['firstenergy', 'first energy'],
    'FFIV': ['f5', 'f5 networks'],
    'FI': ['fiserv'],
    'FICO': ['fico', 'fair isaac'],
    'FIS': ['fis', 'fidelity national information'],
    'FITB': ['fifth third', 'fifth third bank'],
    'FOXA': ['fox', 'fox corporation', 'fox news'],
    'FRT': ['federal realty'],
    'FSLR': ['first solar'],
    'FTNT': ['fortinet'],
    'FTV': ['fortive'],
    
    # === G ===
    'GD': ['general dynamics'],
    'GDDY': ['godaddy', 'go daddy'],
    'GE': ['general electric', 'ge'],
    'GEHC': ['ge healthcare', 'ge healthtech'],
    'GEN': ['gen digital', 'norton', 'lifelock'],
    'GEV': ['ge vernova', 'ge aerospace'],
    'GILD': ['gilead', 'gilead sciences'],
    'GIS': ['general mills'],
    'GL': ['globe life'],
    'GLW': ['corning'],
    'GM': ['general motors', 'gm', 'chevy', 'chevrolet', 'cadillac', 'buick', 'gmc'],
    'GNRC': ['generac'],
    'GOOG': ['google', 'alphabet', 'youtube', 'android', 'waymo', 'deepmind'],
    'GPC': ['genuine parts', 'napa auto parts'],
    'GPN': ['global payments'],
    'GRMN': ['garmin'],
    'GS': ['goldman', 'goldman sachs'],
    'GWW': ['grainger', 'w.w. grainger'],
    
    # === H ===
    'HAL': ['halliburton'],
    'HAS': ['hasbro'],
    'HBAN': ['huntington', 'huntington bancshares', 'huntington bank'],
    'HCA': ['hca', 'hca healthcare'],
    'HD': ['home depot'],
    'HIG': ['hartford', 'the hartford'],
    'HII': ['huntington ingalls', 'huntington ingalls industries'],
    'HLT': ['hilton', 'hilton hotels'],
    'HOLX': ['hologic'],
    'HON': ['honeywell'],
    'HOOD': ['robinhood'],
    'HPE': ['hewlett packard enterprise', 'hpe'],
    'HPQ': ['hp', 'hewlett-packard', 'hp inc'],
    'HRL': ['hormel', 'hormel foods', 'spam'],
    'HSIC': ['henry schein'],
    'HST': ['host hotels', 'host hotels & resorts'],
    'HSY': ['hershey', "hershey's"],
    'HUBB': ['hubbell'],
    'HUM': ['humana'],
    'HWM': ['howmet', 'howmet aerospace'],
    
    # === I ===
    'IBKR': ['interactive brokers'],
    'IBM': ['ibm', 'international business machines'],
    'ICE': ['intercontinental exchange', 'ice'],
    'IDXX': ['idexx', 'idexx laboratories'],
    'IEX': ['idex', 'idex corporation'],
    'IFF': ['iff', 'international flavors', 'international flavors & fragrances'],
    'INCY': ['incyte'],
    'INTC': ['intel', 'core i', 'xeon'],
    'INTU': ['intuit', 'turbotax', 'quickbooks', 'credit karma'],
    'INVH': ['invitation homes'],
    'IP': ['international paper'],
    'IPG': ['interpublic', 'interpublic group'],
    'IQV': ['iqvia'],
    'IR': ['ingersoll rand'],
    'IRM': ['iron mountain'],
    'ISRG': ['intuitive surgical', 'da vinci'],
    'IT': ['gartner'],
    'ITW': ['illinois tool works'],
    'IVZ': ['invesco'],
    
    # === J ===
    'J': ['jacobs', 'jacobs engineering'],
    'JBHT': ['j.b. hunt', 'jb hunt'],
    'JBL': ['jabil'],
    'JCI': ['johnson controls'],
    'JKHY': ['jack henry', 'jack henry & associates'],
    'JNJ': ['johnson & johnson', 'johnson and johnson', 'j&j', 'band-aid', 'tylenol'],
    'JPM': ['jpmorgan', 'jp morgan', 'chase', 'j.p. morgan'],
    
    # === K ===
    'K': ['kellanova', 'kellogg'],
    'KDP': ['keurig dr pepper', 'keurig', 'dr pepper'],
    'KEY': ['keycorp', 'key bank', 'keybank'],
    'KEYS': ['keysight', 'keysight technologies'],
    'KHC': ['kraft heinz', 'kraft', 'heinz'],
    'KIM': ['kimco', 'kimco realty'],
    'KKR': ['kkr', 'kohlberg kravis'],
    'KLAC': ['kla', 'kla corporation'],
    'KMB': ['kimberly-clark', 'kimberly clark', 'kleenex', 'huggies'],
    'KMI': ['kinder morgan'],
    'KO': ['coca-cola', 'coca cola', 'coke'],
    'KR': ['kroger'],
    'KVUE': ['kenvue'],
    
    # === L ===
    'L': ['loews', 'loews corporation'],
    'LDOS': ['leidos'],
    'LEN': ['lennar'],
    'LH': ['labcorp', 'laboratory corporation'],
    'LHX': ['l3harris', 'l3 harris'],
    'LII': ['lennox', 'lennox international'],
    'LIN': ['linde'],
    'LKQ': ['lkq', 'lkq corporation'],
    'LLY': ['eli lilly', 'lilly'],
    'LMT': ['lockheed', 'lockheed martin'],
    'LNT': ['alliant energy'],
    'LOW': ['lowes', "lowe's"],
    'LRCX': ['lam research'],
    'LULU': ['lululemon'],
    'LUV': ['southwest', 'southwest airlines'],
    'LVS': ['las vegas sands', 'sands'],
    'LW': ['lamb weston'],
    'LYB': ['lyondellbasell'],
    'LYV': ['live nation'],
    
    # === M ===
    'MA': ['mastercard', 'master card'],
    'MAA': ['mid-america apartment', 'mid america apartment'],
    'MAR': ['marriott', 'marriott international'],
    'MAS': ['masco'],
    'MCD': ['mcdonald', "mcdonald's", 'mcdonalds'],
    'MCHP': ['microchip', 'microchip technology'],
    'MCK': ['mckesson'],
    'MCO': ['moodys', "moody's"],
    'MDLZ': ['mondelez', 'oreo', 'cadbury'],
    'MDT': ['medtronic'],
    'MET': ['metlife'],
    'META': ['meta', 'facebook', 'instagram', 'whatsapp', 'threads', 'oculus'],
    'MGM': ['mgm', 'mgm resorts'],
    'MHK': ['mohawk', 'mohawk industries'],
    'MKC': ['mccormick'],
    'MLM': ['martin marietta'],
    'MMC': ['marsh mclennan', 'marsh & mclennan'],
    'MMM': ['3m'],
    'MNST': ['monster', 'monster beverage', 'monster energy'],
    'MO': ['altria', 'marlboro', 'philip morris domestic'],
    'MOH': ['molina', 'molina healthcare'],
    'MOS': ['mosaic', 'the mosaic company'],
    'MPC': ['marathon petroleum'],
    'MPWR': ['monolithic power', 'monolithic power systems'],
    'MRK': ['merck'],
    'MRNA': ['moderna'],
    'MS': ['morgan stanley'],
    'MSCI': ['msci'],
    'MSFT': ['microsoft', 'windows', 'azure', 'xbox', 'office 365', 'teams', 'linkedin'],
    'MSI': ['motorola', 'motorola solutions'],
    'MTB': ['m&t bank', 'mt bank'],
    'MTCH': ['match group', 'tinder', 'hinge', 'okcupid'],
    'MTD': ['mettler-toledo', 'mettler toledo'],
    'MU': ['micron', 'micron technology'],
    
    # === N ===
    'NCLH': ['norwegian cruise', 'norwegian cruise line'],
    'NDAQ': ['nasdaq'],
    'NDSN': ['nordson'],
    'NEE': ['nextera', 'nextera energy'],
    'NEM': ['newmont', 'newmont mining'],
    'NFLX': ['netflix'],
    'NI': ['nisource'],
    'NKE': ['nike'],
    'NOC': ['northrop grumman', 'northrop'],
    'NOW': ['servicenow', 'service now'],
    'NRG': ['nrg', 'nrg energy'],
    'NSC': ['norfolk southern'],
    'NTAP': ['netapp', 'net app'],
    'NTRS': ['northern trust'],
    'NUE': ['nucor'],
    'NVDA': ['nvidia', 'geforce', 'rtx', 'cuda', 'tegra'],
    'NVR': ['nvr', 'ryan homes'],
    'NWS': ['news corp', 'news corporation'],
    'NWSA': ['news corp', 'news corporation'],
    'NXPI': ['nxp', 'nxp semiconductors'],
    
    # === O ===
    'O': ['realty income'],
    'ODFL': ['old dominion', 'old dominion freight'],
    'OKE': ['oneok'],
    'OMC': ['omnicom', 'omnicom group'],
    'ON': ['on semiconductor', 'onsemi'],
    'ORCL': ['oracle'],
    'ORLY': ["o'reilly", 'oreilly', "o'reilly auto"],
    'OTIS': ['otis', 'otis worldwide', 'otis elevator'],
    'OXY': ['occidental', 'occidental petroleum'],
    
    # === P ===
    'PANW': ['palo alto networks', 'palo alto'],
    'PAYC': ['paycom'],
    'PAYX': ['paychex'],
    'PCAR': ['paccar', 'kenworth', 'peterbilt'],
    'PCG': ['pg&e', 'pacific gas', 'pacific gas and electric'],
    'PEG': ['pseg', 'public service enterprise'],
    'PEP': ['pepsi', 'pepsico', 'frito-lay', 'frito lay', 'gatorade', 'tropicana'],
    'PFE': ['pfizer'],
    'PFG': ['principal financial', 'principal'],
    'PG': ['procter', 'procter & gamble', 'p&g', 'tide', 'gillette', 'pampers'],
    'PGR': ['progressive', 'progressive insurance'],
    'PH': ['parker hannifin', 'parker-hannifin'],
    'PHM': ['pultegroup', 'pulte homes'],
    'PKG': ['packaging corporation', 'packaging corp of america'],
    'PLD': ['prologis'],
    'PLTR': ['palantir'],
    'PM': ['philip morris', 'philip morris international'],
    'PNC': ['pnc', 'pnc financial', 'pnc bank'],
    'PNR': ['pentair'],
    'PNW': ['pinnacle west'],
    'PODD': ['insulet', 'omnipod'],
    'POOL': ['pool corporation', 'pool corp'],
    'PPG': ['ppg', 'ppg industries'],
    'PPL': ['ppl', 'ppl corporation'],
    'PRU': ['prudential', 'prudential financial'],
    'PSA': ['public storage'],
    'PSKY': ['penumbra'],
    'PSX': ['phillips 66'],
    'PTC': ['ptc'],
    'PWR': ['quanta services', 'quanta'],
    'PYPL': ['paypal', 'venmo'],
    
    # === Q ===
    'QCOM': ['qualcomm', 'snapdragon'],
    
    # === R ===
    'RCL': ['royal caribbean'],
    'REG': ['regency centers'],
    'REGN': ['regeneron'],
    'RF': ['regions financial', 'regions bank'],
    'RJF': ['raymond james'],
    'RL': ['ralph lauren'],
    'RMD': ['resmed'],
    'ROK': ['rockwell automation', 'rockwell'],
    'ROL': ['rollins'],
    'ROP': ['roper', 'roper technologies'],
    'ROST': ['ross stores', 'ross dress for less'],
    'RSG': ['republic services'],
    'RTX': ['rtx', 'raytheon', 'pratt & whitney'],
    'RVTY': ['revvity'],
    
    # === S ===
    'SBAC': ['sba communications'],
    'SBUX': ['starbucks'],
    'SCHW': ['schwab', 'charles schwab'],
    'SHW': ['sherwin-williams', 'sherwin williams'],
    'SJM': ['smucker', 'j.m. smucker', 'jif'],
    'SLB': ['schlumberger', 'slb'],
    'SMCI': ['super micro', 'supermicro'],
    'SNA': ['snap-on', 'snap on'],
    'SNPS': ['synopsys'],
    'SO': ['southern company', 'southern co'],
    'SOLS': ['solventum'],
    'SOLV': ['solventum'],
    'SPG': ['simon property', 'simon property group'],
    'SPGI': ['s&p global', 'sp global'],
    'SRE': ['sempra', 'sempra energy'],
    'STE': ['steris'],
    'STLD': ['steel dynamics'],
    'STT': ['state street'],
    'STX': ['seagate', 'seagate technology'],
    'STZ': ['constellation brands', 'corona', 'modelo'],
    'SW': ['smurfit westrock'],
    'SWK': ['stanley black & decker', 'stanley black and decker'],
    'SWKS': ['skyworks', 'skyworks solutions'],
    'SYF': ['synchrony', 'synchrony financial'],
    'SYK': ['stryker'],
    'SYY': ['sysco'],
    
    # === T ===
    'T': ['at&t', 'att'],
    'TAP': ['molson coors', 'coors', 'miller'],
    'TDG': ['transdigm'],
    'TDY': ['teledyne', 'teledyne technologies'],
    'TECH': ['bio-techne'],
    'TEL': ['te connectivity'],
    'TER': ['teradyne'],
    'TFC': ['truist', 'truist financial'],
    'TGT': ['target'],
    'TJX': ['tjx', 'tj maxx', 'marshalls', 'homegoods'],
    'TKO': ['tko group', 'wwe', 'ufc'],
    'TMO': ['thermo fisher', 'thermo fisher scientific'],
    'TMUS': ['t-mobile', 'tmobile'],
    'TPL': ['texas pacific land'],
    'TPR': ['tapestry', 'coach', 'kate spade'],
    'TRGP': ['targa resources'],
    'TRMB': ['trimble'],
    'TROW': ['t. rowe price', 't rowe price'],
    'TRV': ['travelers', 'the travelers'],
    'TSCO': ['tractor supply'],
    'TSLA': ['tesla', 'cybertruck', 'model 3', 'model y', 'model s', 'model x', 'powerwall'],
    'TSN': ['tyson', 'tyson foods'],
    'TT': ['trane technologies', 'trane'],
    'TTD': ['trade desk', 'the trade desk'],
    'TTWO': ['take-two', 'take two', 'rockstar games', 'gta', 'grand theft auto', '2k'],
    'TXN': ['texas instruments'],
    'TXT': ['textron'],
    'TYL': ['tyler technologies', 'tyler'],
    
    # === U ===
    'UAL': ['united airlines', 'united'],
    'UBER': ['uber', 'uber eats'],
    'UDR': ['udr', 'united dominion realty'],
    'UHS': ['universal health services'],
    'ULTA': ['ulta', 'ulta beauty'],
    'UNH': ['unitedhealth', 'united health', 'united healthcare', 'optum'],
    'UNP': ['union pacific'],
    'UPS': ['ups', 'united parcel service'],
    'URI': ['united rentals'],
    'USB': ['us bancorp', 'u.s. bank', 'us bank'],
    
    # === V ===
    'V': ['visa'],
    'VICI': ['vici properties'],
    'VLO': ['valero', 'valero energy'],
    'VLTO': ['veralto'],
    'VMC': ['vulcan materials', 'vulcan'],
    'VRSK': ['verisk', 'verisk analytics'],
    'VRSN': ['verisign'],
    'VRTX': ['vertex', 'vertex pharmaceuticals'],
    'VST': ['vistra', 'vistra corp'],
    'VTR': ['ventas'],
    'VTRS': ['viatris'],
    'VZ': ['verizon'],
    
    # === W ===
    'WAB': ['wabtec', 'westinghouse air brake'],
    'WAT': ['waters', 'waters corporation'],
    'WBD': ['warner bros', 'warner bros discovery', 'hbo', 'discovery'],
    'WDAY': ['workday'],
    'WDC': ['western digital'],
    'WEC': ['wec energy', 'wisconsin energy'],
    'WELL': ['welltower'],
    'WFC': ['wells fargo'],
    'WM': ['waste management'],
    'WMB': ['williams', 'williams companies'],
    'WMT': ['walmart', 'wal-mart', "sam's club"],
    'WRB': ['w.r. berkley', 'w r berkley'],
    'WSM': ['williams-sonoma', 'williams sonoma', 'pottery barn', 'west elm'],
    'WST': ['west pharmaceutical', 'west pharma'],
    'WTW': ['willis towers watson'],
    'WY': ['weyerhaeuser'],
    'WYNN': ['wynn', 'wynn resorts'],
    
    # === X ===
    'XEL': ['xcel energy', 'xcel'],
    'XOM': ['exxon', 'exxonmobil', 'exxon mobil'],
    'XYL': ['xylem'],
    'XYZ': ['block', 'square', 'cash app'],
    
    # === Y ===
    'YUM': ['yum', 'yum brands', 'kfc', 'pizza hut', 'taco bell'],
    
    # === Z ===
    'ZBH': ['zimmer biomet', 'zimmer'],
    'ZBRA': ['zebra', 'zebra technologies'],
    'ZTS': ['zoetis'],
}


def is_article_relevant(article: dict, ticker: str) -> bool:
    """
    Check if article is relevant to ticker.
    Requires: (1) ticker first in 'related' field, (2) ticker/company/CEO in headline.
    """
    headline = article.get('headline', '').lower()
    related = article.get('related', '')
    
    related_list = [t.strip().upper() for t in related.split(',') if t.strip()]
    
    # REQUIREMENT 1: Ticker must be FIRST in related (primary subject)
    if not related_list or related_list[0] != ticker.upper():
        return False
    
    # REQUIREMENT 2: Must also have keyword match in headline
    # Check 2a: Does headline contain ticker symbol?
    if ticker.lower() in headline:
        return True
    
    # Check 2b: Does headline contain company name/alias?
    aliases = COMPANY_ALIASES.get(ticker.upper(), [])
    for alias in aliases:
        if alias in headline:
            return True
    
    # Check 2c: Does headline contain CEO/executive name for this ticker?
    for ceo_name, ceo_ticker in CEO_TO_TICKER.items():
        if ceo_ticker == ticker.upper() and ceo_name in headline:
            return True
    
    # Ticker was first in related, but no keyword match - reject
    return False


def get_db_connection():
    """Get PostgreSQL connection"""
    postgres_url = os.getenv('DATABASE_URL')

    if not postgres_url:
        log("ERROR: DATABASE_URL not set!")
        sys.exit(1)

    # Fix postgres:// to postgresql://
    if postgres_url.startswith('postgres://'):
        postgres_url = postgres_url.replace('postgres://', 'postgresql://', 1)

    return psycopg2.connect(postgres_url)


def get_all_tickers(conn):
    """Get all tickers from database"""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM daily_prices ORDER BY ticker")
    tickers = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return tickers


class FinBERTAnalyzer:
    """FinBERT sentiment analyzer"""

    def __init__(self):
        log("Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        log(f"FinBERT loaded on {self.device}")

    def analyze(self, text, ticker: str = None):
        """
        Analyze sentiment using FinBERT.
        Returns (score, confidence, label) where score is -1 to +1.
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)

            # FinBERT outputs: [positive, negative, neutral]
            probs = probabilities[0].cpu().numpy()
            positive_prob = float(probs[0])
            negative_prob = float(probs[1])
            neutral_prob = float(probs[2])

            # Calculate sentiment score: positive - negative (range -1 to +1)
            score = positive_prob - negative_prob

            # Confidence is max probability
            confidence = float(max(probs))
            
            max_idx = probs.argmax()
            labels = ['positive', 'negative', 'neutral']
            label = labels[max_idx]

            return score, confidence, label

        except Exception as e:
            log(f"    FinBERT error: {e}")
            return 0.0, 0.5, 'neutral'  # Neutral with low confidence on error


class ClaudeSentimentAnalyzer:
    """Claude Haiku sentiment analyzer"""
    
    def __init__(self, model: str = None):
        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
        self.model = model or CLAUDE_MODEL
        log(f"Claude sentiment analyzer initialized with model: {self.model}")
    
    def analyze(self, text: str, ticker: str = None) -> tuple:
        """
        Analyze sentiment.
        Returns (score, confidence, label) where score is -1 to +1.
        """
        if not text or not text.strip():
            return 0.0, 0.0, "neutral"
        
        prompt = self._build_prompt(text, ticker)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return self._parse_response(response.content[0].text)
        except Exception as e:
            log(f"    Claude API error: {e}")
            return 0.0, 0.5, "neutral"
    
    def _build_prompt(self, text: str, ticker: str = None) -> str:
        """Build sentiment analysis prompt"""
        
        ticker_clause = ""
        if ticker:
            ticker_clause = f"Focus on sentiment as it impacts {ticker} specifically. "
        
        return f"""Analyze the financial sentiment of this text.

{ticker_clause}Return a JSON object with:
- "sentiment": "positive", "neutral", or "negative"
- "score": float from -1.0 (very negative) to 1.0 (very positive)
- "confidence": float from 0.0 to 1.0

TEXT: "{text}"

JSON:"""

    def _parse_response(self, response_text: str) -> tuple:
        """Parse JSON response"""
        try:
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
                
                label = data.get("sentiment", "neutral").lower()
                score = max(-1.0, min(1.0, float(data.get("score", 0.0))))
                confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
                
                if label not in ("positive", "neutral", "negative"):
                    label = "neutral"
                
                return score, confidence, label
        except (json.JSONDecodeError, ValueError) as e:
            log(f"    Claude parse error: {e}")
        
        return 0.0, 0.5, "neutral"


def fetch_sentiment(conn, tickers, analyzer):
    """
    Fetch news from Finnhub and analyze sentiment.
    Returns set of tickers that received new articles.
    """
    api_key = os.getenv('FINNHUB_API_KEY')

    if not api_key:
        log("FINNHUB_API_KEY not set - skipping sentiment fetch")
        return set()

    log(f"Fetching sentiment for {len(tickers)} tickers...")

    cursor = conn.cursor()

    # Ticker aliases: fetch news from alias and save under main ticker
    TICKER_ALIASES = {
        'GOOG': ['GOOGL'],  # Also fetch GOOGL news for GOOG
    }

    total_articles = 0
    saved_articles = 0
    filtered_articles = 0 
    failed_tickers = []
    ticker_count = len(tickers)
    tickers_with_new_articles = set() 

    for idx, ticker in enumerate(tickers):
        # Progress every 50 tickers
        if idx % 50 == 0:
            log(f"  Progress: {idx}/{ticker_count} tickers processed")

        # Get symbols to fetch news for (main ticker + any aliases)
        symbols_to_fetch = [ticker] + TICKER_ALIASES.get(ticker, [])

        for fetch_symbol in symbols_to_fetch:
            try:
                # Finnhub news endpoint - fetch last 2 days
                from_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
                to_date = datetime.now().strftime('%Y-%m-%d')

                url = "https://finnhub.io/api/v1/company-news"
                params = {
                    'symbol': fetch_symbol,
                    'from': from_date,
                    'to': to_date,
                    'token': api_key
                }

                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    articles = response.json()

                    if articles:
                        total_articles += len(articles)
                        ticker_saved = 0
                        ticker_filtered = 0

                        for article in articles:
                            try:
                                headline = article.get('headline', '')
                                summary = article.get('summary', '')
                                article_url = article.get('url', '')
                                source = article.get('source', '')
                                published_ts = article.get('datetime', 0)

                                if published_ts:
                                    published_at = datetime.fromtimestamp(published_ts)
                                else:
                                    published_at = datetime.now()

                                if not headline or not article_url:
                                    continue

                                # Check if article is relevant to this ticker
                                if not is_article_relevant(article, ticker):
                                    ticker_filtered += 1
                                    filtered_articles += 1
                                    continue

                                cursor.execute("""
                                    INSERT INTO news_articles
                                    (ticker, headline, summary, url, source, published_at, fetched_at)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (url) DO NOTHING
                                    RETURNING id
                                """, (ticker, headline, summary, article_url, source, published_at, datetime.now()))

                                result = cursor.fetchone()
                                
                                if not result:
                                    continue
                                    
                                article_id = result[0]

                                sentiment_score, confidence, sentiment_label = analyzer.analyze(headline, ticker=ticker)

                                cursor.execute("""
                                    INSERT INTO sentiment_scores
                                    (news_article_id, ticker, sentiment_score, confidence, sentiment_label, analyzed_at)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                """, (article_id, ticker, sentiment_score, confidence, sentiment_label, datetime.now()))

                                ticker_saved += 1
                                saved_articles += 1
                                tickers_with_new_articles.add(ticker)

                            except Exception as e:
                                log(f"    Error saving article for {ticker}: {e}")
                                conn.rollback()
                                continue

                        if ticker_saved > 0 or ticker_filtered > 0:
                            log(f"  {ticker}: {ticker_saved} saved, {ticker_filtered} filtered as irrelevant (from {fetch_symbol})")
                        elif len(articles) > 0:
                            log(f"  {ticker}: {len(articles)} articles from {fetch_symbol} (all duplicates)")

                        conn.commit()

                # API LIMIT 60 / MIN !!! (overshoot for safety)
                time.sleep(1.1)

            except Exception as e:
                failed_tickers.append(ticker)
                log(f"  Error fetching {fetch_symbol} for {ticker}: {e}")
                continue

    cursor.close()

    log(f"Sentiment fetch completed: {total_articles} fetched, {saved_articles} saved, {filtered_articles} filtered as irrelevant")
    log(f"Tickers with new articles: {len(tickers_with_new_articles)}")
    if failed_tickers:
        log(f"Failed: {len(failed_tickers)} tickers")

    return tickers_with_new_articles  # Return set of tickers that need aggreg


def compute_weighted_sentiment(conn, ticker, lambda_decay=0.25, lookback_days=7):
    """
    Compute time-weighted sentiment score and update summary table.
    Uses exponential decay (lambda=0.25) over 7-day window.
    """
    import math
    
    cursor = conn.cursor()
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    
    # Fetch all articles with sentiment scores
    cursor.execute("""
        SELECT 
            ss.sentiment_score,
            ss.confidence,
            ss.sentiment_label,
            na.published_at
        FROM sentiment_scores ss
        JOIN news_articles na ON ss.news_article_id = na.id
        WHERE ss.ticker = %s
          AND na.published_at >= %s
        ORDER BY na.published_at DESC
    """, (ticker, cutoff_date))
    
    rows = cursor.fetchall()
    
    if not rows:
        cursor.close()
        return None
    
    # Calculate weighted sentiment
    now = datetime.now()
    weighted_sum = 0.0
    weight_sum = 0.0
    simple_sum = 0.0
    
    positive_count = 0
    neutral_count = 0
    negative_count = 0
    total_confidence = 0.0
    
    oldest_date = None
    newest_date = None
    
    for row in rows:
        score = float(row[0])
        confidence = float(row[1])
        label = row[2]
        published_at = row[3]
        
        # Track dates
        if oldest_date is None or published_at < oldest_date:
            oldest_date = published_at
        if newest_date is None or published_at > newest_date:
            newest_date = published_at
        
        time_diff = (now - published_at).total_seconds() / 86400
        
        temporal_decay = math.exp(-lambda_decay * time_diff)
        weight = confidence * temporal_decay
        
        weighted_sum += weight * score
        weight_sum += weight
        simple_sum += score
        total_confidence += confidence
        
        if label == 'positive':
            positive_count += 1
        elif label == 'negative':
            negative_count += 1
        else:
            neutral_count += 1
    
    article_count = len(rows)
    
    weighted_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    simple_avg_score = simple_sum / article_count
    avg_confidence = total_confidence / article_count
    
    positive_pct = (positive_count / article_count) * 100
    neutral_pct = (neutral_count / article_count) * 100
    negative_pct = (negative_count / article_count) * 100
    
    # Upsert into ticker_sentiment_summary
    cursor.execute("""
        INSERT INTO ticker_sentiment_summary (
            ticker,
            weighted_sentiment_score,
            simple_avg_score,
            article_count,
            positive_count,
            neutral_count,
            negative_count,
            avg_confidence,
            oldest_article_date,
            newest_article_date,
            lambda_decay,
            lookback_days,
            last_updated,
            positive_pct,
            neutral_pct,
            negative_pct
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker) DO UPDATE SET
            weighted_sentiment_score = EXCLUDED.weighted_sentiment_score,
            simple_avg_score = EXCLUDED.simple_avg_score,
            article_count = EXCLUDED.article_count,
            positive_count = EXCLUDED.positive_count,
            neutral_count = EXCLUDED.neutral_count,
            negative_count = EXCLUDED.negative_count,
            avg_confidence = EXCLUDED.avg_confidence,
            oldest_article_date = EXCLUDED.oldest_article_date,
            newest_article_date = EXCLUDED.newest_article_date,
            lambda_decay = EXCLUDED.lambda_decay,
            lookback_days = EXCLUDED.lookback_days,
            last_updated = EXCLUDED.last_updated,
            positive_pct = EXCLUDED.positive_pct,
            neutral_pct = EXCLUDED.neutral_pct,
            negative_pct = EXCLUDED.negative_pct
    """, (
        ticker,
        weighted_score,
        simple_avg_score,
        article_count,
        positive_count,
        neutral_count,
        negative_count,
        avg_confidence,
        oldest_date,
        newest_date,
        lambda_decay,
        lookback_days,
        datetime.now(),
        positive_pct,
        neutral_pct,
        negative_pct
    ))
    
    cursor.close()
    conn.commit()
    
    return {
        'ticker': ticker,
        'weighted_score': weighted_score,
        'simple_avg_score': simple_avg_score,
        'article_count': article_count
    }


def update_sentiment_summaries(conn, tickers_to_update):
    """Update sentiment summaries for tickers with new articles"""
    if not tickers_to_update:
        log("No tickers to update (no new articles)")
        return
        
    log(f"Updating weighted sentiment summaries for {len(tickers_to_update)} tickers with new articles...")
    
    updated = 0
    for ticker in tickers_to_update:
        result = compute_weighted_sentiment(conn, ticker)
        if result:
            updated += 1
    
    log(f"Updated {updated} ticker sentiment summaries")


def main():
    """Main execution"""
    log("=" * 60)
    analyzer_name = "Claude Haiku" if USE_CLAUDE_SENTIMENT else "FinBERT"
    log(f"Starting sentiment fetch job ({analyzer_name})")
    log("=" * 60)

    try:
        # Initialize sentiment analyzer based on configuration
        if USE_CLAUDE_SENTIMENT:
            analyzer = ClaudeSentimentAnalyzer()
        else:
            analyzer = FinBERTAnalyzer()

        conn = get_db_connection()
        log("Connected to PostgreSQL")

        tickers = get_all_tickers(conn)
        log(f"Found {len(tickers)} tickers in database")

        # Fetch and analyze new articles
        tickers_with_new_articles = fetch_sentiment(conn, tickers, analyzer)
        
        # Update weighted sentiment summaries for all tickers
        update_sentiment_summaries(conn, tickers)
        conn.close()

        log("=" * 60)
        log("Sentiment fetch job completed successfully")
        log("=" * 60)

        sys.exit(0)

    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
