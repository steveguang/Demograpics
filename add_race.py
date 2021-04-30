
def getRemainHandles(path):
    f = open(path)
    f.readline()
    handles = set()
    for line in f:
        line = line.strip()
        handle = line.split("\x1b")[0]
        handles.add(handle)
    f.close()
    return handles

def map_race(filename):
    races = {}
    f = open(filename)
    whites = [race.lower() for race in "white,irish,Jewish,English,American,Armenians,Armenian,Albanians,Serbs,Greeks,Italian,Swedish-speaking,British,Scottish,Swedish,Poles,Pashtuns,Ukrainians,Kurds,German,Hungarians,Germans,Scotch-irish,Georgians,Russians,Swedes,Bulgarians,Italians,Ashkenazi,Ukrainian,Iranian,Austrians,Welsh,Czechs,Canadian,Albanian,Norwegians,Danes,Slovak,Polish,Transylvanian,Danish,Persian,Romanians,Icelanders,Australians,Croatian,Jews,Spaniards,French,Ossetians,Macedonians,Belarusians,Serbs|scottish,Romanian,Puerto,Australian,Cajun,Stateside,Arab,Swiss,Portuguese,Yazidis,Croats".split(",")]
    asians = [race.lower() for race in "Bengali,Japanese,Koreans,Indian,Korean,Punjabi,Tamil,han,Thai,Chinese,Malaysian,Indians,Dravida,Hoklo,Monguor,Filipino,Telugu,shan,Tibetan,Turkish,Malayali,Koreans|korean,Taiwanese,Pakistanis|pakistani,Sinhala,Vietnamese,Assamese,Pakistanis,Asian,Nepalis".split(",")]
    black = [race.lower() for race in "African, Afro-Americans,Yoruba,Fula,Black,Igbo,Baganda,Haitian,Somalis,Kiga,Zulu,Malians,Tutsi,Nyakyusa,Nigerian,Oromo,sotho".split(",")]
    for line in f:
        info = line.strip().split("\x1b")
        handle, race = info[1].lower(), info[-1].lower()
        if not race:
            continue
        race = race.split(" ")[0]
        if race in whites:
            races[handle] = "white"
        elif race in black:
            races[handle] = "black"
        elif race in asians:
            races[handle] = "asian"
        else:
            races[handle] = "other"
    return races

def get_races(filename):
    f = open(filename)
    f.readline()
    genders_ages = {}
    for line in f:
        line = line.strip()
        name, age, gender, handle, race, verified = line.split("\x1b")
        if not race:
            continue
        if race == "african":
            genders_ages[handle.lower()] = ["1"]
        else:
            genders_ages[handle.lower()] = ["0"]
    return genders_ages

def write_file(r_path, w, is_filter_handles):
    f = open(r_path)
    for line in f:
        info = line.strip().split(",")[:-2]
        handle = info[0].lower()
        if is_filter_handles and (handle not in handles):
            continue
        if handle in races:
            val = "1" if races[handle] == "black" else "0"
            w.write(",".join(info+[val])+"\n")
    f.close()

handles = getRemainHandles("new_nonstop_onefeaturesword1.csv")
races = map_race("query_race_attributes.csv")
path = "transdb_bert-uncased-nli"
w = open(path+"/all_features_black_nonblack_more_race.csv", 'w')

f = open(path+"/mergedAll.csv")
w.write(",".join(f.readline().strip().split(",")[:-2]+['race'])+"\n")
f.close()

write_file(path+"/mergedAll.csv", w, True)
write_file(path+"/mergedAllMore.csv", w, False)
w.close()
