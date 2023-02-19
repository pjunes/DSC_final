import FileManager
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import pytextrank
from nltk.tokenize import sent_tokenize
from pajek_tools import PajekWriter

###########
# Prob. 1 #
###########

#################
# Article 생성 전 #
#################

# article_labeling
# str 상태의 article_raw 를 가공
# 주요 목적은 여러줄로 나누어져 있는 AB 등의 값을 한줄로 만드는것

# input article_raw
# output article_labeled : list - ["PMID- ...", "DP  - ...", ... ]
def article_labeling(article_raw):
    article_labeled = list()
    article_lines = article_raw.split("\n")

    # 처음 6글자가 공백이면 기존영역
    # 처음 6글자가 "....- " 이면 새로운 영역
    header = str()
    for line in article_lines:

        if line[:6] == "      ":
            # 기존 영역에 속하므로 헤더에 line 더하기.
            # 줄맞춤을 위해 있던 6칸짜리 공백 지우기.
            header += line[6:]
        else:
            # 새로운 영역이므로, 전영역이 있었다면 저장 후 새로운 헤더 지정
            if header:
                article_labeled.append(header)
            header = line

    return article_labeled

def article_dicting(article_raw):
    article_labeled = article_labeling(article_raw)
    article_dict = dict()
    for item in article_labeled:
        # item[:4].strip() : "AB  " -> "AB"
        # item[:4] item[6:] 사이 잘리는 부분은 "- "
        key, value = item[:4].strip(), item[6:]
        # key가 이미 존재한다면, [기존 value] + [새로운 value] = [기존1, 기존2, 새로운]
        # key가 존재하지 않았다면, [] + [새로운 value] = [새로운]
        article_dict[key] = article_dict.get(key, list()) + [value]
    return article_dict

############
# 존재 테스트 #
############

# PMID DP TI AB 를 복수로 가지는 article 은 없으며,
# AB가 없는 논문 2개, TI가 없는 논문 6개, 둘다 없는 논문은 없다.
# AD, AU, MH 가 없는 논문은 많다.
# cnt = 0
# for Article in Article_list:
#     if len(Article.PMID) != 1:
#         print(f"PMID {len(Article.PMID)}")
#
#     if len(Article.DP) != 1:
#         print(f"DP   {len(Article.DP)}")
#
#     if Article.TI:
#         if len(Article.TI) != 1:
#             print(f"TI   {len(Article.TI)}")
#     else:
#         print(f"TI   None")
#         if not Article.AB:
#             print("both")
#
#     if Article.AB:
#         if len(Article.AB) != 1:
#             print(f"AB   {len(Article.AB)}")
#     else:
#         print("AB   None")
#
#     if not Article.AU:
#         print("AU None")
#     if not Article.AD:
#         print("AD None")
#     if not Article.MH:
#         print("MH None")
#
#
#     if (Article.AB):
#         if (Article.TI):
#             cnt += 1
#
# print(cnt) # 7645 + 2 + 6 = 7653

def article_dict_processing(article_dict):
    article_dict_processed = dict()
    # 필요한 7개의 값만 처리
    # PMID DP TI AB AU AD MH
    # 항상 존재하며 1개만 존재 (1)
    article_dict_processed["PMID"] = article_dict["PMID"][0]
    article_dict_processed["DP"] = article_dict["DP"][0]
    # 1개이거나 존재하지 않을 수 있는 값들. (0~1)
    article_dict_processed["TI"] = article_dict.get("TI")
    article_dict_processed["AB"] = article_dict.get("AB")
    # 복수로 존재할 수도 있는 값들. (0~ )
    article_dict_processed["AU"] = article_dict.get("AU")
    article_dict_processed["AD"] = article_dict.get("AD")
    article_dict_processed["MH"] = article_dict.get("MH")

    return article_dict_processed

###########
# Article #
###########

class Article:

    # Article의 개수
    counter = 0

    def __init__(self, article_raw):
        Article.counter += 1
        self.data = article_dicting(article_raw)

        # article_dict_processing()는 모든 라벨 처리
        # Article의 생성자는 문제에 필요한 7개의 값만 처리
        # PMID DP TI AB AU AD MH

        # 항상 존재하며 1개만 존재 (1)
        self.PMID = self.data["PMID"][0]
        self.DP = self.data["DP"][0]
        # 1개이거나 존재하지 않을 수 있는 값들. (0~1)
        self.TI = self.data.get("TI", [None])[0]
        self.AB = self.data.get("AB", [None])[0]
        # 복수로 존재할 수도 있는 값들. (0~ )
        self.AU = self.data.get("AU")
        self.AD = self.data.get("AD")
        self.MH = self.data.get("MH")

    def __str__(self):
        return f"PMID : {self.PMID} (Title : {self.TI})"

    def get_DP(self):
        DP = re.findall("([A-Z][a-z]{2})", self.DP)
        if DP:
            return DP[-1]
        return None

    def get_tag(self, tag):
        return self.data.get(tag)

    # def real_DP(self):
    #     DP = self.get_DP()
    #     if (not DP) | (DP == "Sup") | (DP == "Spr") | (DP == "Sum") | (DP == "Qua"):
    #         return None
    #     return DP


#######################################################################################################
#############                                     main                                   ##############
#######################################################################################################

print("##### Prob 1 #####")

# file_data : str - 파일을 통쨰로 read() 로 읽어옴.
file_data = FileManager.read_file("final_report.txt")

# data_sep : list - "\n\n" 로 구분되어 있는 논문들을 각각 리스트에 담음.
data_sep = re.split("\n\n",file_data)
# print(len(data_sep)) # 논문의 개수 7653

Article_list = list()

# article_raw : str - 논문 1개에 대한 날것의 정보 ( 아무 과정을 거치지 않음 )
for article_raw in data_sep:
    Article_list.append(Article(article_raw))

print(f"클래스 메서드를 정의하여 Article 객체의 수를 관리"
      f"\n객체의 수(list len()) : {len(Article_list)}\n"
      f"객체의 수(cls method) : {Article.counter}")
print()

print(f"print 함수로 Article 객체를 출력하는 형식\n"
      f"print(Article) => "
      f"{Article_list[0]}")
print()

print(f"태그의 값을 획득할 수 있는 기능\n"
      f"Article.get_tag(\"LR\") => "
      f"{Article_list[0].get_tag('LR')}")
print(f"Article.get_tag(\"DP\") => "
      f"{Article_list[0].get_tag('AU')}")

print()
###########
# Prob. 2 #
###########
print("##### Prob 2 #####")

# DP_dict = [{"None":0},{"Jan":5},{"Feb":3}, ...]
DP_dict = {}
for Article in Article_list:
    DP = Article.get_DP()
    # print(f"{Article.DP} / {DP}")
    DP_dict[DP] = DP_dict.get(DP, 0) + 1

print(DP_dict)
DP_dict[None] += DP_dict.pop("Sup") + DP_dict.pop("Spr") + DP_dict.pop("Sum") + DP_dict.pop("Qua")
# 예외 : None / 'Sup' / 'Spring' / 'Summer' / '...-Quarter'
# 전부 None으로 통합.

sum = 0
for num in DP_dict.values():
    sum += num
print(f"processed : {sum - DP_dict[None]}\n"
      f"exception : {DP_dict[None]}\n"
      f"Total : {Article.counter}")

Month_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# DP_list = [DP_dict["Jan"], DP_dict["Feb"], DP_dict["Mar"], DP_dict["Apr"], DP_dict["May"], DP_dict["Jun"]
#     , DP_dict["Jul"], DP_dict["Aug"], DP_dict["Sep"], DP_dict["Oct"], DP_dict["Nov"], DP_dict["Dec"]]

DP_list = [DP_dict[month] for month in Month_list]

print(pd.DataFrame(DP_list, Month_list, columns = ["count"]))

plt.plot(Month_list ,DP_list)
plt.xlabel("Month")
plt.ylabel("Count")
plt.title("DP by Month")

####################### 문제 2번 주석 해제!!! #########################
# plt.show()

print()
###########
# Prob. 3 #
###########
print("##### Prob 3 #####")

def get_country_top10(Article_list):
    country_dict = {}
    cnt = 0
    for Article in Article_list:
        if Article.AD:
            AD_last = Article.AD[-1]
            # print(AD_last)

            data_list = re.findall(", ([^\.\,@]*?)\.", Article.AD[-1])
            # if not data_list:
            #     data_list = re.findall(", ([^\.\,@]*?);", Article.AD[-1])
            if not data_list:
                # ", Country" 유형
                data_list = re.findall(", ([^\.\,@]*?)$", Article.AD[-1])
                if not data_list:
                    # ", Country email" 유형
                    data_list = re.findall(", ([^\.\,@]*?) [a-zA-Z0-9]+@[a-zA-Z0-9.]+", Article.AD[-1])
                    if not data_list:
                        # " Country;" 유형
                        data_list = re.findall(" ([^\.\,@]*?);", Article.AD[-1])

            if not data_list:
                cnt += 1
                continue

            country = data_list[-1]

            # 나라가 아닌 University 제외
            # if re.search("Univ", country):
            #     cnt += 1
            #     continue
            if "Univ" in country:
                cnt += 1
                continue

            # if len(country) == 1:
            #     print(f"{country} : ", end = "")
            #     print(AD_last)
            # elif country == "":
            #     print("item : ", end = "")
            #     print()

            country_dict[country] = country_dict.get(country, 0) + 1
        else:  # Article.AD == None
            cnt += 1

    # sum = 0
    # for num in country_dict.values():
    #     sum += num
    # print(f"processed : {sum}\n"
    #       f"exception : {cnt}\n"
    #       f"total : {cnt + sum}")

    # 예외처리
    try:
        country_dict["Republic of Korea"] = country_dict.get("Republic of Korea", 0) + country_dict.pop("South Korea") + country_dict.pop("Korea")
    except:
        pass
    try:
        country_dict["China"] = country_dict.get("China", 0) + country_dict.pop("P") + country_dict.pop("PR China") + country_dict.pop("ROC")
    except:
        pass
    # 의문점? 현재 가지고 있는 데이터로는 country가 "P"가 나오면 무조건 P.R. China 에서 나온 결과라 모두 China로 계산되게 처리했습니다.
    # 하지만 새로운 데이터가 추가되었을때도 "P"가 China라는 보장이 없는데 어떻게 해야하는지 의문이 들었습니다.
    try:
        country_dict["China"] = country_dict.get("China", 0) + country_dict.pop("People's Republic of China")
    except:
        pass
    try:
        country_dict["United States of America"] = country_dict.get("United States of America", 0) + country_dict.pop("United Sates of America") + country_dict.pop(
        "United States")
    except:
        pass
    try:
        country_dict["United States of America"] = country_dict.get("United States of America", 0) + country_dict.pop("USA") + country_dict.pop("CA") + country_dict.pop(
        "NY")
    except:
        pass
    try:
        country_dict["United States of America"] = country_dict.get("United States of America", 0) + country_dict.pop("Minneapolis") + country_dict.pop("Maryland")
    except:
        pass
    try:
        country_dict["United States of America"] = country_dict.get("United States of America", 0) + country_dict.pop("New York") + country_dict.pop("Florida")
    except:
        pass
    try:
        country_dict["United States of America"] = country_dict.get("United States of America", 0) + country_dict.pop("California") + country_dict.pop("Pennsylvania")
    except:
        pass
    try:
        country_dict["United States of America"] = country_dict.get("United States of America", 0) + country_dict.pop("Illinois") + country_dict.pop("Ohio")
    except:
        pass
    try:
        country_dict["Texas"] = country_dict.get("Texas", 0) + country_dict.pop("Tex") + country_dict.pop("TEXAS")
    except:
        pass
    try:
        country_dict["United Kingdom"] = country_dict.get("United Kingdom", 0) + country_dict.pop("UK") + country_dict.pop("Reino Unido")
    except:
        pass
    try:
        country_dict["Switzerland"] = country_dict.get("Switzerland", 0) + country_dict.pop("1011 Lausanne") + country_dict.pop("Schweiz")
    except:
        pass
    try:
        country_dict["Brazil"] = country_dict.get("Brazil", 0) + country_dict.pop("BR") + country_dict.pop("Brasil")
    except:
        pass
    try:
        country_dict["Netherlands"] = country_dict.get("Netherlands", 0) + country_dict.pop("the Netherlands") + country_dict.pop("The Netherlands")
    except:
        pass

    # country_dict[] += country_dict.pop()
    # country_dict[] += country_dict.pop()

    available_country = dict()
    na_country = dict()

    for key in country_dict.keys():
        if country_dict[key] > 1:  # 1개 이하 걸러냄. / TOP 10을 선정하는데 문제 없을것.
            available_country[key] = country_dict[key]
        else:  # if country_dict[key] == 1:
            na_country[key] = country_dict[key]
            cnt += country_dict[key]

    for key in na_country:
        if "USA" in key:
            available_country["United States of America"] = available_country.get("United States of America", 0) + na_country[key]
            cnt -= na_country[key]
            continue
        elif ("Korea" in key) & (key != "North Korea"):
            available_country["Republic of Korea"] = available_country.get("Republic of Korea", 0) + na_country[key]
            cnt -= na_country[key]
            continue
        elif "China" in key:
            available_country["China"] = available_country.get("China", 0) + na_country[key]
            cnt -= na_country[key]
            continue

        for country in available_country.keys():
            if country in key:
                available_country[country] += na_country[key]
                cnt -= na_country[key]

    available_country_cnt = Counter(available_country)
    return available_country_cnt.most_common(10) # top10_country

def sep_by_DP(Article_list):
    sep_by_DP = [[],[],[],[]]
    exception_cnt = 0 # 예외 카운트
    processed_cnt = 0 # 처리 카운트
    for Article in Article_list:
        DP = Article.get_DP()
        i = -1
        if DP in Month_list[:3]:
            i = 0
        elif DP in Month_list[3:6]:
            i = 1
        elif DP in Month_list[6:9]:
            i = 2
        elif DP in Month_list[9:]:
            i = 3
        else:
            exception_cnt += 1

        if not i < 0:
            sep_by_DP[i].append(Article)
            # print(i)
            processed_cnt += 1

    if len(Article_list) != exception_cnt + processed_cnt:
        print("cnt error")
        exit()
    print()
    return sep_by_DP

sep_by_DP = sep_by_DP(Article_list)
index = [i for i in range(1, 11)]
columns = ["Country", "counter"]
# columns = ["Q1", "Q2", "Q3", "Q4"]

cnt = 0
for Article_by_quarter in sep_by_DP:
    cnt += 1
    print(f"========== Quarter{cnt} Table ==========")
    top10_country = get_country_top10(Article_by_quarter)
    DP_top10_by_quarter_table = pd.DataFrame(map(list, top10_country), index, columns)
    print(DP_top10_by_quarter_table)



# country_dict = {}
# cnt = 0
# for Article in Article_list:
#     if Article.AD:
#         AD_last = Article.AD[-1]
#         # print(AD_last)
#
#         data_list = re.findall(", ([^\.\,@]*?)\.", Article.AD[-1])
#         # if not data_list:
#         #     data_list = re.findall(", ([^\.\,@]*?);", Article.AD[-1])
#         if not data_list:
#             # ", Country" 유형
#             data_list = re.findall(", ([^\.\,@]*?)$", Article.AD[-1])
#             if not data_list:
#                 # ", Country email" 유형
#                 data_list = re.findall(", ([^\.\,@]*?) [a-zA-Z0-9]+@[a-zA-Z0-9.]+", Article.AD[-1])
#                 if not data_list:
#                     # " Country;" 유형
#                     data_list =  re.findall(" ([^\.\,@]*?);", Article.AD[-1])
#
#         if not data_list:
#             cnt += 1
#             continue
#
#         country = data_list[-1]
#
#         # 나라가 아닌 University 제외
#         # if re.search("Univ", country):
#         #     cnt += 1
#         #     continue
#         if "Univ" in country:
#             cnt += 1
#             continue
#
#         # if len(country) == 1:
#         #     print(f"{country} : ", end = "")
#         #     print(AD_last)
#         # elif country == "":
#         #     print("item : ", end = "")
#         #     print()
#
#         country_dict[country] = country_dict.get(country, 0) + 1
#     else: # Article.AD == None
#         cnt += 1
#
# # sum = 0
# # for num in country_dict.values():
# #     sum += num
# # print(f"processed : {sum}\n"
# #       f"exception : {cnt}\n"
# #       f"total : {cnt + sum}")
#
# # 예외처리
# country_dict["Republic of Korea"] += country_dict.pop("South Korea") + country_dict.pop("Korea")
# country_dict["China"] += country_dict.pop("P") + country_dict.pop("PR China") + country_dict.pop("ROC")
# # 의문점? 현재 가지고 있는 데이터로는 country가 "P"가 나오면 무조건 P.R. China 에서 나온 결과라 모두 China로 계산되게 처리했습니다.
# # 하지만 새로운 데이터가 추가되었을때도 "P"가 China라는 보장이 없는데 어떻게 해야하는지 의문이 들었습니다.
# country_dict["China"] += country_dict.pop("People's Republic of China")
# country_dict["United States of America"] += country_dict.pop("United Sates of America") + country_dict.pop("United States")
# country_dict["United States of America"] += country_dict.pop("USA") + country_dict.pop("CA") + country_dict.pop("NY")
# country_dict["United States of America"] += country_dict.pop("Minneapolis") + country_dict.pop("Maryland")
# country_dict["United States of America"] += country_dict.pop("New York") + country_dict.pop("Florida")
# country_dict["United States of America"] += country_dict.pop("California") + country_dict.pop("Pennsylvania")
# country_dict["United States of America"] += country_dict.pop("Illinois") + country_dict.pop("Ohio")
# country_dict["Texas"] += country_dict.pop("Tex") + country_dict.pop("TEXAS")
# country_dict["United Kingdom"] += country_dict.pop("UK") + country_dict.pop("Reino Unido")
# country_dict["Switzerland"] += country_dict.pop("1011 Lausanne") + country_dict.pop("Schweiz")
# country_dict["Brazil"] += country_dict.pop("BR") + country_dict.pop("Brasil")
# country_dict["Netherlands"] += country_dict.pop("the Netherlands") + country_dict.pop("The Netherlands")
#
# # country_dict[] += country_dict.pop()
# # country_dict[] += country_dict.pop()
#
# available_country = dict()
# na_country = dict()
#
# for key in country_dict.keys():
#     if country_dict[key] > 10: # 10개 이하 걸러냄. / TOP 10을 선정하는데 문제 없을것.
#         available_country[key] = country_dict[key]
#     else: # if country_dict[key] == 1:
#         na_country[key] = country_dict[key]
#         cnt += country_dict[key]
#
# for key in na_country:
#     if "USA" in key:
#         available_country["United States of America"] += na_country[key]
#         cnt -= na_country[key]
#         continue
#     elif ("Korea" in key) & (key != "North Korea"):
#         available_country["Republic of Korea"] += na_country[key]
#         cnt -= na_country[key]
#         continue
#     elif "China" in key:
#         available_country["China"] += na_country[key]
#         cnt -= na_country[key]
#         continue
#
#     for country in available_country.keys():
#         if country in key:
#             available_country[country] += na_country[key]
#             cnt -= na_country[key]
#
#
# available_country_cnt = Counter(available_country)
# top10_country = available_country_cnt.most_common(10)
#
# sum = 0
# for num in available_country.values():
#     sum += num
# print()
# print(f"processed : {sum}")
# print(f"exception : {cnt}")
# print(f"total : {sum + cnt}")
#
#
# index = [i for i in range(1, 11)]
# columns = ["Country", "count"]
# DP_top10_table = pd.DataFrame(top10_country, index, columns)
# print(DP_top10_table)

print()
###########
# Prob. 4 #
###########
print("##### Prob 4 #####")

keyword_freq = dict()
for Article in Article_list:
    if not Article.MH:
        continue
    for word in Article.MH:
        # for word in word.split("/"):
        word = word.split("/")[-1].strip("*")
        keyword_freq[word] = keyword_freq.get(word, 0) + 1
keyword_cnt = Counter(keyword_freq)
prob4 = keyword_cnt.most_common(100)

index = [i for i in range(1, 101)]
columns = ["keyword", "count"]
MH_top10_table = pd.DataFrame(prob4, index, columns)
pd.set_option('display.max_row', None)
print(MH_top10_table)

print()
##############
# Prob. plus #
##############
print("##### Prob plus #####")

del sum

def get_similarity(freq1, freq2):
    # 두개의 매개변수의 크기가 다를경우 similarity를 계산 할 수 없으므로 오류.
    if len(freq1) != len(freq2):
        return None

    # 분모 : 두 값의 최솟값들의 합
    # 분자 : 두 값의 최댓값들의 합

    items = list(zip(freq1, freq2))
    upper = sum(min(item) for item in items)
    lower = sum(max(item) for item in items)

    if lower == 0: return 0

    return upper / lower

nlp = spacy.load("en_core_web_sm")
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

def get_doc(Article):
    doc = nlp(Article.AB) # AB 들어갈 자리
    return doc

keywords_list = []
similar_keywords = list()

###################################################### 여기서 부터 10분 소요 ####
for Article in Article_list:
    if not Article.AB:
        continue
    doc = get_doc(Article)
    keywords = list()
    cnt = 0
    for p in doc._.phrases:
        cnt += 1
        # print(f"{p.rank:.4f} {p.count:5d} {p.text}")
        if p.rank >= 0.06:
            keywords.append(p.text)
    keywords_list.append(keywords)

    sents = [sent for sent in sent_tokenize(Article.AB)]
    keyword_freqs = list()
    for keyword in keywords:
        keyword_freq = [0] * len(sents)
        for i in range(len(sents)):
            if keyword in sents[i]:
                keyword_freq[i] = 1
        keyword_freqs.append(keyword_freq)

    # print(keyword_freqs)
    for i in range(len(keywords) - 1):
        for j in range(i+1, len(keywords)):
            similarity = get_similarity(keyword_freqs[i], keyword_freqs[j])
            if similarity > 0.4:
                similar_keywords.append([keywords[i], keywords[j]])
# vertex = dict()
# vertex_id = 1
# for keyword1, keyword2 in similar_keywords:
#     if keyword1 not in vertex:
#         vertex[keyword1] = vertex_id
#         vertex_id += 1
#     if keyword2 not in vertex:
#         vertex[keyword2] = vertex_id
#         vertex_id += 1
print("유사도가 0.4 초과인 키워드 쌍")
print(similar_keywords)
# df = pd.DataFrame(similar_keywords, columns=["key1", "key2"])
# writer = PajekWriter(similar_keywords, directed = False, cited_colname = "key1", citing_colname = "key2")
# writer.write("keyword.net")







