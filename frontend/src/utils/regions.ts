/**
 * TDWG World Geographical Scheme for Recording Plant Distributions (WGSRPD)
 * Based on: Brummitt, R.K. (2001). World Geographical Scheme for Recording
 * Plant Distributions, Edition 2. TDWG, Hunt Institute for Botanical
 * Documentation, Pittsburgh.
 */

export interface TDWGRegion {
  code: string;
  name: string;
  level: 1 | 2 | 3;
  parent?: string; // L1 code for L2; L2 code for L3
}

export const L1 = [
  { code: "1", name: "Europe", level: 1 },
  { code: "2", name: "Africa", level: 1 },
  { code: "3", name: "Asia-Temperate", level: 1 },
  { code: "4", name: "Asia-Tropical", level: 1 },
  { code: "5", name: "Australasia", level: 1 },
  { code: "6", name: "Pacific", level: 1 },
  { code: "7", name: "Northern America", level: 1 },
  { code: "8", name: "Southern America", level: 1 },
  { code: "9", name: "Antarctic", level: 1 },
] as const satisfies TDWGRegion[];

export const L2 = [
  // 1 Europe
  { code: "10", name: "Northern Europe", level: 2, parent: "1" },
  { code: "11", name: "Middle Europe", level: 2, parent: "1" },
  { code: "12", name: "Southwestern Europe", level: 2, parent: "1" },
  { code: "13", name: "Southeastern Europe", level: 2, parent: "1" },
  { code: "14", name: "Eastern Europe", level: 2, parent: "1" },
  // 2 Africa
  { code: "20", name: "Northern Africa", level: 2, parent: "2" },
  { code: "21", name: "Macaronesia", level: 2, parent: "2" },
  { code: "22", name: "West Tropical Africa", level: 2, parent: "2" },
  { code: "23", name: "West-Central Tropical Africa", level: 2, parent: "2" },
  { code: "24", name: "Northeast Tropical Africa", level: 2, parent: "2" },
  { code: "25", name: "East Tropical Africa", level: 2, parent: "2" },
  { code: "26", name: "South Tropical Africa", level: 2, parent: "2" },
  { code: "27", name: "Southern Africa", level: 2, parent: "2" },
  { code: "28", name: "Middle Atlantic Ocean", level: 2, parent: "2" },
  { code: "29", name: "Western Indian Ocean", level: 2, parent: "2" },
  // 3 Asia-Temperate
  { code: "30", name: "Siberia", level: 2, parent: "3" },
  { code: "31", name: "Russian Far East", level: 2, parent: "3" },
  { code: "32", name: "Middle Asia", level: 2, parent: "3" },
  { code: "33", name: "Caucasus", level: 2, parent: "3" },
  { code: "34", name: "Western Asia", level: 2, parent: "3" },
  { code: "35", name: "Arabian Peninsula", level: 2, parent: "3" },
  { code: "36", name: "China", level: 2, parent: "3" },
  { code: "37", name: "Mongolia", level: 2, parent: "3" },
  { code: "38", name: "Eastern Asia", level: 2, parent: "3" },
  // 4 Asia-Tropical
  { code: "40", name: "Indian Subcontinent", level: 2, parent: "4" },
  { code: "41", name: "Indo-China", level: 2, parent: "4" },
  { code: "42", name: "Malesia", level: 2, parent: "4" },
  { code: "43", name: "Papuasia", level: 2, parent: "4" },
  // 5 Australasia
  { code: "50", name: "Australia", level: 2, parent: "5" },
  { code: "51", name: "New Zealand", level: 2, parent: "5" },
  // 6 Pacific
  { code: "60", name: "Southwestern Pacific", level: 2, parent: "6" },
  { code: "61", name: "South-Central Pacific", level: 2, parent: "6" },
  { code: "62", name: "Northwestern Pacific", level: 2, parent: "6" },
  { code: "63", name: "North-Central Pacific", level: 2, parent: "6" },
  // 7 Northern America
  { code: "70", name: "Subarctic America", level: 2, parent: "7" },
  { code: "71", name: "Western Canada", level: 2, parent: "7" },
  { code: "72", name: "Eastern Canada", level: 2, parent: "7" },
  { code: "73", name: "Northwestern U.S.A.", level: 2, parent: "7" },
  { code: "74", name: "North-Central U.S.A.", level: 2, parent: "7" },
  { code: "75", name: "Northeastern U.S.A.", level: 2, parent: "7" },
  { code: "76", name: "Southwestern U.S.A.", level: 2, parent: "7" },
  { code: "77", name: "South-Central U.S.A.", level: 2, parent: "7" },
  { code: "78", name: "Southeastern U.S.A.", level: 2, parent: "7" },
  { code: "79", name: "Mexico", level: 2, parent: "7" },
  // 8 Southern America
  { code: "80", name: "Central America", level: 2, parent: "8" },
  { code: "81", name: "Caribbean", level: 2, parent: "8" },
  { code: "82", name: "Northern South America", level: 2, parent: "8" },
  { code: "83", name: "Western South America", level: 2, parent: "8" },
  { code: "84", name: "Brazil", level: 2, parent: "8" },
  { code: "85", name: "Southern South America", level: 2, parent: "8" },
  // 9 Antarctic
  { code: "90", name: "Subantarctic Islands", level: 2, parent: "9" },
  { code: "91", name: "Antarctic Continent", level: 2, parent: "9" },
] as const satisfies TDWGRegion[];

export const L3 = [
  // 10 Northern Europe
  { code: "DEN", name: "Denmark", level: 3, parent: "10" },
  { code: "FIN", name: "Finland", level: 3, parent: "10" },
  { code: "FRO", name: "Føroyar", level: 3, parent: "10" },
  { code: "GRB", name: "Great Britain", level: 3, parent: "10" },
  { code: "ICE", name: "Iceland", level: 3, parent: "10" },
  { code: "IRE", name: "Ireland", level: 3, parent: "10" },
  { code: "NOR", name: "Norway", level: 3, parent: "10" },
  { code: "SVA", name: "Svalbard", level: 3, parent: "10" },
  { code: "SWE", name: "Sweden", level: 3, parent: "10" },
  // 11 Middle Europe
  { code: "AUT", name: "Austria", level: 3, parent: "11" },
  { code: "BGM", name: "Belgium", level: 3, parent: "11" },
  { code: "CZE", name: "Czechoslovakia", level: 3, parent: "11" },
  { code: "GER", name: "Germany", level: 3, parent: "11" },
  { code: "HUN", name: "Hungary", level: 3, parent: "11" },
  { code: "NET", name: "Netherlands", level: 3, parent: "11" },
  { code: "POL", name: "Poland", level: 3, parent: "11" },
  { code: "SWI", name: "Switzerland", level: 3, parent: "11" },
  // 12 Southwestern Europe
  { code: "BAL", name: "Baleares", level: 3, parent: "12" },
  { code: "COR", name: "Corse", level: 3, parent: "12" },
  { code: "FRA", name: "France", level: 3, parent: "12" },
  { code: "POR", name: "Portugal", level: 3, parent: "12" },
  { code: "SAR", name: "Sardegna", level: 3, parent: "12" },
  { code: "SPA", name: "Spain", level: 3, parent: "12" },
  // 13 Southeastern Europe
  { code: "ALB", name: "Albania", level: 3, parent: "13" },
  { code: "BUL", name: "Bulgaria", level: 3, parent: "13" },
  { code: "GRC", name: "Greece", level: 3, parent: "13" },
  { code: "ITA", name: "Italy", level: 3, parent: "13" },
  { code: "KRI", name: "Kriti", level: 3, parent: "13" },
  { code: "ROM", name: "Romania", level: 3, parent: "13" },
  { code: "SIC", name: "Sicilia", level: 3, parent: "13" },
  { code: "TUE", name: "Turkey-in-Europe", level: 3, parent: "13" },
  { code: "YUG", name: "Yugoslavia", level: 3, parent: "13" },
  // 14 Eastern Europe
  { code: "BLR", name: "Belarus", level: 3, parent: "14" },
  { code: "BLT", name: "Baltic States", level: 3, parent: "14" },
  { code: "KRY", name: "Krym", level: 3, parent: "14" },
  { code: "RUC", name: "Central European Russia", level: 3, parent: "14" },
  { code: "RUE", name: "East European Russia", level: 3, parent: "14" },
  { code: "RUN", name: "North European Russia", level: 3, parent: "14" },
  { code: "RUS", name: "South European Russia", level: 3, parent: "14" },
  { code: "RUW", name: "Northwest European Russia", level: 3, parent: "14" },
  { code: "UKR", name: "Ukraine", level: 3, parent: "14" },
  // 20 Northern Africa
  { code: "ALG", name: "Algeria", level: 3, parent: "20" },
  { code: "EGY", name: "Egypt", level: 3, parent: "20" },
  { code: "LBY", name: "Libya", level: 3, parent: "20" },
  { code: "MOR", name: "Morocco", level: 3, parent: "20" },
  { code: "TUN", name: "Tunisia", level: 3, parent: "20" },
  { code: "WSA", name: "Western Sahara", level: 3, parent: "20" },
  // 21 Macaronesia
  { code: "AZO", name: "Azores", level: 3, parent: "21" },
  { code: "CNY", name: "Canary Is.", level: 3, parent: "21" },
  { code: "CVI", name: "Cape Verde", level: 3, parent: "21" },
  { code: "MDR", name: "Madeira", level: 3, parent: "21" },
  { code: "SEL", name: "Selvagens", level: 3, parent: "21" },
  // 22 West Tropical Africa
  { code: "BEN", name: "Benin", level: 3, parent: "22" },
  { code: "BKN", name: "Burkina", level: 3, parent: "22" },
  { code: "GAM", name: "Gambia, The", level: 3, parent: "22" },
  { code: "GHA", name: "Ghana", level: 3, parent: "22" },
  { code: "GNB", name: "Guinea-Bissau", level: 3, parent: "22" },
  { code: "GUI", name: "Guinea", level: 3, parent: "22" },
  { code: "IVO", name: "Ivory Coast", level: 3, parent: "22" },
  { code: "LBR", name: "Liberia", level: 3, parent: "22" },
  { code: "MLI", name: "Mali", level: 3, parent: "22" },
  { code: "MTN", name: "Mauritania", level: 3, parent: "22" },
  { code: "NGA", name: "Nigeria", level: 3, parent: "22" },
  { code: "NGR", name: "Niger", level: 3, parent: "22" },
  { code: "SEN", name: "Senegal", level: 3, parent: "22" },
  { code: "SIE", name: "Sierra Leone", level: 3, parent: "22" },
  { code: "TOG", name: "Togo", level: 3, parent: "22" },
  // 23 West-Central Tropical Africa
  { code: "BUR", name: "Burundi", level: 3, parent: "23" },
  { code: "CAB", name: "Cabinda", level: 3, parent: "23" },
  { code: "CAF", name: "Central African Republic", level: 3, parent: "23" },
  { code: "CMN", name: "Cameroon", level: 3, parent: "23" },
  { code: "CGO", name: "Congo", level: 3, parent: "23" },
  { code: "EQG", name: "Equatorial Guinea", level: 3, parent: "23" },
  { code: "GAB", name: "Gabon", level: 3, parent: "23" },
  { code: "GGI", name: "Gulf of Guinea Is.", level: 3, parent: "23" },
  { code: "RWA", name: "Rwanda", level: 3, parent: "23" },
  { code: "ZAI", name: "Zaire", level: 3, parent: "23" },
  // 24 Northeast Tropical Africa
  { code: "CHA", name: "Chad", level: 3, parent: "24" },
  { code: "DJI", name: "Djibouti", level: 3, parent: "24" },
  { code: "ERI", name: "Eritrea", level: 3, parent: "24" },
  { code: "ETH", name: "Ethiopia", level: 3, parent: "24" },
  { code: "SOC", name: "Socotra", level: 3, parent: "24" },
  { code: "SOM", name: "Somalia", level: 3, parent: "24" },
  { code: "SUD", name: "Sudan", level: 3, parent: "24" },
  // 25 East Tropical Africa
  { code: "KEN", name: "Kenya", level: 3, parent: "25" },
  { code: "TAN", name: "Tanzania", level: 3, parent: "25" },
  { code: "UGA", name: "Uganda", level: 3, parent: "25" },
  // 26 South Tropical Africa
  { code: "ANG", name: "Angola", level: 3, parent: "26" },
  { code: "MLW", name: "Malawi", level: 3, parent: "26" },
  { code: "MOZ", name: "Mozambique", level: 3, parent: "26" },
  { code: "ZAM", name: "Zambia", level: 3, parent: "26" },
  { code: "ZIM", name: "Zimbabwe", level: 3, parent: "26" },
  // 27 Southern Africa
  { code: "BOT", name: "Botswana", level: 3, parent: "27" },
  { code: "CPP", name: "Cape Provinces", level: 3, parent: "27" },
  { code: "CPV", name: "Caprivi Strip", level: 3, parent: "27" },
  { code: "LES", name: "Lesotho", level: 3, parent: "27" },
  { code: "NAM", name: "Namibia", level: 3, parent: "27" },
  { code: "NAT", name: "KwaZulu-Natal", level: 3, parent: "27" },
  { code: "OFS", name: "Free State", level: 3, parent: "27" },
  { code: "SWZ", name: "Swaziland", level: 3, parent: "27" },
  { code: "TVL", name: "Northern Provinces", level: 3, parent: "27" },
  // 28 Middle Atlantic Ocean
  { code: "ASC", name: "Ascension", level: 3, parent: "28" },
  { code: "STH", name: "St. Helena", level: 3, parent: "28" },
  { code: "TRI", name: "Trindade", level: 3, parent: "28" },
  // 29 Western Indian Ocean
  { code: "ALD", name: "Aldabra", level: 3, parent: "29" },
  { code: "CGS", name: "Chagos Archipelago", level: 3, parent: "29" },
  { code: "COM", name: "Comoros", level: 3, parent: "29" },
  { code: "MAU", name: "Mauritius", level: 3, parent: "29" },
  { code: "MCI", name: "Mozambique Channel Is.", level: 3, parent: "29" },
  { code: "MDG", name: "Madagascar", level: 3, parent: "29" },
  { code: "REU", name: "Réunion", level: 3, parent: "29" },
  { code: "ROD", name: "Rodrigues", level: 3, parent: "29" },
  { code: "SEY", name: "Seychelles", level: 3, parent: "29" },
  // 30 Siberia
  { code: "ALT", name: "Altay", level: 3, parent: "30" },
  { code: "BRY", name: "Buryatiya", level: 3, parent: "30" },
  { code: "CTA", name: "Chita", level: 3, parent: "30" },
  { code: "IRK", name: "Irkutsk", level: 3, parent: "30" },
  { code: "KRA", name: "Krasnoyarsk", level: 3, parent: "30" },
  { code: "TVA", name: "Tuva", level: 3, parent: "30" },
  { code: "WSB", name: "West Siberia", level: 3, parent: "30" },
  { code: "YAK", name: "Yakutskiya", level: 3, parent: "30" },
  // 31 Russian Far East
  { code: "AMU", name: "Amur", level: 3, parent: "31" },
  { code: "KAM", name: "Kamchatka", level: 3, parent: "31" },
  { code: "KHA", name: "Khabarovsk", level: 3, parent: "31" },
  { code: "KUR", name: "Kuril Is.", level: 3, parent: "31" },
  { code: "MAG", name: "Magadan", level: 3, parent: "31" },
  { code: "PRM", name: "Primorye", level: 3, parent: "31" },
  { code: "SAK", name: "Sakhalin", level: 3, parent: "31" },
  // 32 Middle Asia
  { code: "KAZ", name: "Kazakhstan", level: 3, parent: "32" },
  { code: "KGZ", name: "Kirgizistan", level: 3, parent: "32" },
  { code: "TKM", name: "Turkmenistan", level: 3, parent: "32" },
  { code: "TZK", name: "Tadzhikistan", level: 3, parent: "32" },
  { code: "UZB", name: "Uzbekistan", level: 3, parent: "32" },
  // 33 Caucasus
  { code: "NCS", name: "North Caucasus", level: 3, parent: "33" },
  { code: "TCS", name: "Transcaucasus", level: 3, parent: "33" },
  // 34 Western Asia
  { code: "AFG", name: "Afghanistan", level: 3, parent: "34" },
  { code: "CYP", name: "Cyprus", level: 3, parent: "34" },
  { code: "EAI", name: "East Aegean Is.", level: 3, parent: "34" },
  { code: "IRN", name: "Iran", level: 3, parent: "34" },
  { code: "IRQ", name: "Iraq", level: 3, parent: "34" },
  { code: "LBS", name: "Lebanon-Syria", level: 3, parent: "34" },
  { code: "PAL", name: "Palestine", level: 3, parent: "34" },
  { code: "SIN", name: "Sinai", level: 3, parent: "34" },
  { code: "TUR", name: "Turkey", level: 3, parent: "34" },
  // 35 Arabian Peninsula
  { code: "GST", name: "Gulf States", level: 3, parent: "35" },
  { code: "KUW", name: "Kuwait", level: 3, parent: "35" },
  { code: "OMA", name: "Oman", level: 3, parent: "35" },
  { code: "SAU", name: "Saudi Arabia", level: 3, parent: "35" },
  { code: "YEM", name: "Yemen", level: 3, parent: "35" },
  // 36 China
  { code: "CHC", name: "China South-Central", level: 3, parent: "36" },
  { code: "CHH", name: "Hainan", level: 3, parent: "36" },
  { code: "CHI", name: "Inner Mongolia", level: 3, parent: "36" },
  { code: "CHM", name: "Manchuria", level: 3, parent: "36" },
  { code: "CHN", name: "China North-Central", level: 3, parent: "36" },
  { code: "CHQ", name: "Qinghai", level: 3, parent: "36" },
  { code: "CHS", name: "China Southeast", level: 3, parent: "36" },
  { code: "CHT", name: "Tibet", level: 3, parent: "36" },
  { code: "CHX", name: "Xinjiang", level: 3, parent: "36" },
  // 37 Mongolia
  { code: "MON", name: "Mongolia", level: 3, parent: "37" },
  // 38 Eastern Asia
  { code: "JAP", name: "Japan", level: 3, parent: "38" },
  { code: "KOR", name: "Korea", level: 3, parent: "38" },
  { code: "KZN", name: "Kazan-retto", level: 3, parent: "38" },
  { code: "NNS", name: "Nansei-shoto", level: 3, parent: "38" },
  { code: "OGA", name: "Ogasawara-shoto", level: 3, parent: "38" },
  { code: "TAI", name: "Taiwan", level: 3, parent: "38" },
  // 40 Indian Subcontinent
  { code: "BAN", name: "Bangladesh", level: 3, parent: "40" },
  { code: "EHM", name: "East Himalaya", level: 3, parent: "40" },
  { code: "IND", name: "India", level: 3, parent: "40" },
  { code: "MDV", name: "Maldives", level: 3, parent: "40" },
  { code: "NEP", name: "Nepal", level: 3, parent: "40" },
  { code: "PAK", name: "Pakistan", level: 3, parent: "40" },
  { code: "SRL", name: "Sri Lanka", level: 3, parent: "40" },
  { code: "WHM", name: "West Himalaya", level: 3, parent: "40" },
  // 41 Indo-China
  { code: "AND", name: "Andaman Is.", level: 3, parent: "41" },
  { code: "CBD", name: "Cambodia", level: 3, parent: "41" },
  { code: "LAO", name: "Laos", level: 3, parent: "41" },
  { code: "MYA", name: "Myanmar", level: 3, parent: "41" },
  { code: "NCB", name: "Nicobar Is.", level: 3, parent: "41" },
  { code: "THA", name: "Thailand", level: 3, parent: "41" },
  { code: "VIE", name: "Vietnam", level: 3, parent: "41" },
  // 42 Malesia
  { code: "BOR", name: "Borneo", level: 3, parent: "42" },
  { code: "JAW", name: "Jawa", level: 3, parent: "42" },
  { code: "LSI", name: "Lesser Sunda Is.", level: 3, parent: "42" },
  { code: "MLY", name: "Malaya", level: 3, parent: "42" },
  { code: "MOL", name: "Maluku", level: 3, parent: "42" },
  { code: "PHI", name: "Philippines", level: 3, parent: "42" },
  { code: "SUL", name: "Sulawesi", level: 3, parent: "42" },
  { code: "SUM", name: "Sumatera", level: 3, parent: "42" },
  // 43 Papuasia
  { code: "BIS", name: "Bismarck Archipelago", level: 3, parent: "43" },
  { code: "NWG", name: "New Guinea", level: 3, parent: "43" },
  { code: "SOL", name: "Solomon Is.", level: 3, parent: "43" },
  // 50 Australia
  { code: "NSW", name: "New South Wales", level: 3, parent: "50" },
  { code: "NTA", name: "Northern Territory", level: 3, parent: "50" },
  { code: "QLD", name: "Queensland", level: 3, parent: "50" },
  { code: "SOA", name: "South Australia", level: 3, parent: "50" },
  { code: "TAS", name: "Tasmania", level: 3, parent: "50" },
  { code: "VIC", name: "Victoria", level: 3, parent: "50" },
  { code: "WAU", name: "Western Australia", level: 3, parent: "50" },
  // 51 New Zealand
  { code: "NZN", name: "New Zealand North", level: 3, parent: "51" },
  { code: "NZS", name: "New Zealand South", level: 3, parent: "51" },
  // 60 Southwestern Pacific
  { code: "FIJ", name: "Fiji", level: 3, parent: "60" },
  { code: "NCA", name: "New Caledonia", level: 3, parent: "60" },
  { code: "VAN", name: "Vanuatu", level: 3, parent: "60" },
  // 61 South-Central Pacific
  { code: "COO", name: "Cook Is.", level: 3, parent: "61" },
  { code: "NUE", name: "Niue", level: 3, parent: "61" },
  { code: "SAM", name: "Samoa", level: 3, parent: "61" },
  { code: "TON", name: "Tonga", level: 3, parent: "61" },
  { code: "TUA", name: "Tuamotu", level: 3, parent: "61" },
  { code: "WAL", name: "Wallis-Futuna Is.", level: 3, parent: "61" },
  // 62 Northwestern Pacific
  { code: "GIL", name: "Gilbert Is.", level: 3, parent: "62" },
  { code: "MCS", name: "Micronesia", level: 3, parent: "62" },
  { code: "NRU", name: "Nauru", level: 3, parent: "62" },
  // 63 North-Central Pacific
  { code: "HAW", name: "Hawaiian Is.", level: 3, parent: "63" },
  { code: "LIN", name: "Line Is.", level: 3, parent: "63" },
  { code: "MRQ", name: "Marquesas", level: 3, parent: "63" },
  // 70 Subarctic America
  { code: "ALE", name: "Aleutian Is.", level: 3, parent: "70" },
  { code: "ASK", name: "Alaska", level: 3, parent: "70" },
  { code: "GNL", name: "Greenland", level: 3, parent: "70" },
  { code: "NUN", name: "Nunavut", level: 3, parent: "70" },
  { code: "NWT", name: "Northwest Territories", level: 3, parent: "70" },
  { code: "YUK", name: "Yukon", level: 3, parent: "70" },
  // 71 Western Canada
  { code: "ABT", name: "Alberta", level: 3, parent: "71" },
  { code: "BRC", name: "British Columbia", level: 3, parent: "71" },
  { code: "MAN", name: "Manitoba", level: 3, parent: "71" },
  { code: "SAS", name: "Saskatchewan", level: 3, parent: "71" },
  // 72 Eastern Canada
  { code: "LAB", name: "Labrador", level: 3, parent: "72" },
  { code: "NBR", name: "New Brunswick", level: 3, parent: "72" },
  { code: "NFD", name: "Newfoundland", level: 3, parent: "72" },
  { code: "NSC", name: "Nova Scotia", level: 3, parent: "72" },
  { code: "ONT", name: "Ontario", level: 3, parent: "72" },
  { code: "PEI", name: "Prince Edward Is.", level: 3, parent: "72" },
  { code: "QUE", name: "Québec", level: 3, parent: "72" },
  // 73 Northwestern U.S.A.
  { code: "IDA", name: "Idaho", level: 3, parent: "73" },
  { code: "MNT", name: "Montana", level: 3, parent: "73" },
  { code: "ORE", name: "Oregon", level: 3, parent: "73" },
  { code: "WAS", name: "Washington", level: 3, parent: "73" },
  { code: "WYO", name: "Wyoming", level: 3, parent: "73" },
  // 74 North-Central U.S.A.
  { code: "COL", name: "Colorado", level: 3, parent: "74" },
  { code: "IOW", name: "Iowa", level: 3, parent: "74" },
  { code: "KAN", name: "Kansas", level: 3, parent: "74" },
  { code: "MIN", name: "Minnesota", level: 3, parent: "74" },
  { code: "MIS", name: "Missouri", level: 3, parent: "74" },
  { code: "NEB", name: "Nebraska", level: 3, parent: "74" },
  { code: "NDK", name: "North Dakota", level: 3, parent: "74" },
  { code: "SDK", name: "South Dakota", level: 3, parent: "74" },
  { code: "WIS", name: "Wisconsin", level: 3, parent: "74" },
  // 75 Northeastern U.S.A.
  { code: "CTC", name: "Connecticut", level: 3, parent: "75" },
  { code: "DEL", name: "Delaware", level: 3, parent: "75" },
  { code: "ILL", name: "Illinois", level: 3, parent: "75" },
  { code: "INI", name: "Indiana", level: 3, parent: "75" },
  { code: "MAI", name: "Maine", level: 3, parent: "75" },
  { code: "MAS", name: "Massachusetts", level: 3, parent: "75" },
  { code: "MDY", name: "Maryland", level: 3, parent: "75" },
  { code: "MIC", name: "Michigan", level: 3, parent: "75" },
  { code: "NHN", name: "New Hampshire", level: 3, parent: "75" },
  { code: "NJE", name: "New Jersey", level: 3, parent: "75" },
  { code: "NYK", name: "New York", level: 3, parent: "75" },
  { code: "OHI", name: "Ohio", level: 3, parent: "75" },
  { code: "PEN", name: "Pennsylvania", level: 3, parent: "75" },
  { code: "RHO", name: "Rhode Island", level: 3, parent: "75" },
  { code: "VER", name: "Vermont", level: 3, parent: "75" },
  { code: "VIR", name: "Virginia", level: 3, parent: "75" },
  { code: "WDC", name: "District of Columbia", level: 3, parent: "75" },
  { code: "WVA", name: "West Virginia", level: 3, parent: "75" },
  // 76 Southwestern U.S.A.
  { code: "ARI", name: "Arizona", level: 3, parent: "76" },
  { code: "CAL", name: "California", level: 3, parent: "76" },
  { code: "NEV", name: "Nevada", level: 3, parent: "76" },
  { code: "NWM", name: "New Mexico", level: 3, parent: "76" },
  { code: "UTA", name: "Utah", level: 3, parent: "76" },
  // 77 South-Central U.S.A.
  { code: "ARK", name: "Arkansas", level: 3, parent: "77" },
  { code: "KTY", name: "Kentucky", level: 3, parent: "77" },
  { code: "LOU", name: "Louisiana", level: 3, parent: "77" },
  { code: "MSI", name: "Mississippi", level: 3, parent: "77" },
  { code: "OKL", name: "Oklahoma", level: 3, parent: "77" },
  { code: "TEN", name: "Tennessee", level: 3, parent: "77" },
  { code: "TEX", name: "Texas", level: 3, parent: "77" },
  // 78 Southeastern U.S.A.
  { code: "ALA", name: "Alabama", level: 3, parent: "78" },
  { code: "FLA", name: "Florida", level: 3, parent: "78" },
  { code: "GEO", name: "Georgia", level: 3, parent: "78" },
  { code: "NCS", name: "North Carolina", level: 3, parent: "78" },
  { code: "SCA", name: "South Carolina", level: 3, parent: "78" },
  // 79 Mexico
  { code: "MEX", name: "Mexico", level: 3, parent: "79" },
  // 80 Central America
  { code: "BLZ", name: "Belize", level: 3, parent: "80" },
  { code: "COS", name: "Costa Rica", level: 3, parent: "80" },
  { code: "ELS", name: "El Salvador", level: 3, parent: "80" },
  { code: "GUA", name: "Guatemala", level: 3, parent: "80" },
  { code: "HON", name: "Honduras", level: 3, parent: "80" },
  { code: "NIC", name: "Nicaragua", level: 3, parent: "80" },
  { code: "PAN", name: "Panamá", level: 3, parent: "80" },
  // 81 Caribbean
  { code: "BAH", name: "Bahamas", level: 3, parent: "81" },
  { code: "BER", name: "Bermuda", level: 3, parent: "81" },
  { code: "CAY", name: "Cayman Is.", level: 3, parent: "81" },
  { code: "CUB", name: "Cuba", level: 3, parent: "81" },
  { code: "DOM", name: "Dominican Republic", level: 3, parent: "81" },
  { code: "HAI", name: "Haiti", level: 3, parent: "81" },
  { code: "JAM", name: "Jamaica", level: 3, parent: "81" },
  { code: "LEE", name: "Leeward Is.", level: 3, parent: "81" },
  { code: "PUE", name: "Puerto Rico", level: 3, parent: "81" },
  { code: "TRT", name: "Trinidad-Tobago", level: 3, parent: "81" },
  { code: "WIN", name: "Windward Is.", level: 3, parent: "81" },
  // 82 Northern South America
  { code: "FRG", name: "French Guiana", level: 3, parent: "82" },
  { code: "GUY", name: "Guyana", level: 3, parent: "82" },
  { code: "SUR", name: "Suriname", level: 3, parent: "82" },
  { code: "VEN", name: "Venezuela", level: 3, parent: "82" },
  // 83 Western South America
  { code: "BOL", name: "Bolivia", level: 3, parent: "83" },
  { code: "CLM", name: "Colombia", level: 3, parent: "83" },
  { code: "ECU", name: "Ecuador", level: 3, parent: "83" },
  { code: "GAL", name: "Galápagos", level: 3, parent: "83" },
  { code: "PER", name: "Peru", level: 3, parent: "83" },
  // 84 Brazil
  { code: "BZC", name: "Brazil Central", level: 3, parent: "84" },
  { code: "BZE", name: "Brazil East", level: 3, parent: "84" },
  { code: "BZN", name: "Brazil North", level: 3, parent: "84" },
  { code: "BZL", name: "Brazil Northeast", level: 3, parent: "84" },
  { code: "BZS", name: "Brazil South", level: 3, parent: "84" },
  // 85 Southern South America
  { code: "ARG", name: "Argentina", level: 3, parent: "85" },
  { code: "CHL", name: "Chile", level: 3, parent: "85" },
  { code: "PAR", name: "Paraguay", level: 3, parent: "85" },
  { code: "URU", name: "Uruguay", level: 3, parent: "85" },
  // 90 Subantarctic Islands
  { code: "AMS", name: "Amsterdam-St.Paul Is.", level: 3, parent: "90" },
  { code: "BOU", name: "Bouvet I.", level: 3, parent: "90" },
  { code: "CRZ", name: "Crozet Is.", level: 3, parent: "90" },
  { code: "FLK", name: "Falkland Is.", level: 3, parent: "90" },
  { code: "TDC", name: "Tristan da Cunha", level: 3, parent: "90" },
  { code: "HRD", name: "Heard Is.", level: 3, parent: "90" },
  { code: "KER", name: "Kerguelen Is.", level: 3, parent: "90" },
  { code: "MCQ", name: "Macquarie Is.", level: 3, parent: "90" },
  { code: "MPE", name: "Marion-Prince Edward Is.", level: 3, parent: "90" },
  { code: "SGE", name: "South Georgia", level: 3, parent: "86" },
  // 91 Antarctic Continent
  { code: "ANT", name: "Antarctica", level: 3, parent: "91" },
] as const satisfies TDWGRegion[];

export type RegionCode =
  | (typeof L1)[number]["code"]
  | (typeof L2)[number]["code"]
  | (typeof L3)[number]["code"];

export const allRegions = [...L1, ...L2, ...L3];
export const byCode: Record<string, TDWGRegion> = Object.fromEntries(
  allRegions.map((r) => [r.code, r]),
);
export const l2ByParent = L2.reduce(
  (acc, r) => {
    (acc[r.parent!] ??= []).push(r);
    return acc;
  },
  {} as Record<string, TDWGRegion[]>,
);
export const l3ByParent = L3.reduce(
  (acc, r) => {
    (acc[r.parent!] ??= []).push(r);
    return acc;
  },
  {} as Record<string, TDWGRegion[]>,
);

export function resolveAncestry(code: string) {
  const r = byCode[code];
  if (!r) return {};
  if (r.level === 1) return { l1: r };
  if (r.level === 2) return { l1: byCode[r.parent!], l2: r };
  const l2 = byCode[r.parent!];
  return { l1: l2 ? byCode[l2.parent!] : undefined, l2, l3: r };
}
