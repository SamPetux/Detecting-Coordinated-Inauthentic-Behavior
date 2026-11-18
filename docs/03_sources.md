# Реестр источников

Статусы: ✅ первичный источник открыт (abs/текст); 🔎+ найден поиском, совпадают название/авторы/идентификатор,
первичный текст не открыт; ⚠️ не проверен (по памяти — обязательно проверить до использования); 📄 из ВКР/своих работ.
Перед подачей у каждой цитируемой ссылки — ✅ или проверка автором по DOI. Числа из чужих работ в статью — только
после открытия первоисточника (✅).

## Ядро: детекция координации

| Ключ | Источник | Идентификатор | Статус | Зачем в статье |
|---|---|---|---|---|
| pacheco2021 | D. Pacheco, P.-M. Hui, C. Torres-Lugo, B. T. Truong, A. Flammini, F. Menczer. Uncovering Coordinated Networks on Social Media: Methods and Case Studies. ICWSM 2021, pp. 455–466 | arXiv:2001.05658; DOI 10.1609/icwsm.v15i1.18075 | 🔎+ | Базовая схема «сеть сходства по следам поведения»; код IUNetSci/coordination-detection |
| magelinski2022 | T. Magelinski, L. Ng, K. M. Carley. A Synchronized Action Framework for Detection of Coordination on Social Media. J. Online Trust & Safety, 2022 | arXiv:2105.07454 | 🔎+ | Синхронные действия, мультивидовость; текст сознательно исключён — наше отличие |
| luceri2024 | L. Luceri, V. Pantè, K. Burghardt, E. Ferrara. Unmasking the Web of Deceit: Uncovering Coordinated Activity to Expose Information Operations on Twitter. WWW 2024 | DOI — найти; код github.com/ValeriaPante/coordinatedActivity (MIT) ✅ | 🔎+ | Главный бейзлайн: 5 сетей сходства (fast retweet, co-retweet, co-URL, hashtag sequence, text similarity) + fused network. **Код прочитан ✅ (18.11): text similarity = эмбеддинги `stsb-xlm-r-multilingual`, косинус ≥ 0,7, окно — сутки, без нуль-модели** |
| luceri2024code | V. Pantè et al. coordinatedActivity: code for «Unmasking the Web of Deceit» (MIT) | github.com/ValeriaPante/coordinatedActivity, файл textSimilarity.py | ✅ (код прочитан 18.11.2026) | Текстовый слой Luceri et al.: эмбеддинги stsb-xlm-r-multilingual, косинус ≥ 0,7, окна в сутки, без проверки значимости |
| iannucci2026 | L. Iannucci, E. Muratore, A. Matakos, M. Kivelä. Detecting Coordinated Activities Through Temporal, Multiplex, and Collaborative Analysis. ICWSM 2026 | arXiv:2512.19677 | ✅ (abs+HTML) | Ближайшая работа: мультиплекс действий + экспоненциальное временно́е ядро + Leiden; 26 кампаний Seckin; метрики F1*, NMI, WP; код заявлен github.com/letiziaia/time-aware-collaboration (не проверен) |
| minici2025iohunter | M. Minici, L. Luceri, F. Fabbri, E. Ferrara. IOHunter: Graph Foundation Model to Uncover Online Information Operations. AAAI 2025 | arXiv:2412.14663 | 🔎+ | Текст + граф сходства, обучение с учителем; мы — без учителя |
| mannocci2024survey | L. Mannocci, M. Mazza, A. Monreale, M. Tesconi, S. Cresci. Detection and Characterization of Coordinated Online Behavior: A Survey. 2024 | arXiv:2408.01257 (журнал не подтверждён) | 🔎+ | Определения координации/недостоверности, обзор 84 работ |
| tardelli2024 | S. Tardelli et al. Temporal Dynamics of Coordinated Online Behavior: Stability, Archetypes, and Influence | arXiv:2301.06774 (журнал проверить) | 🔎+ | Временна́я устойчивость координации |
| giglietto2020 | F. Giglietto, N. Righetti, L. Rossi, G. Marino. It takes a village to manipulate the media: coordinated link sharing behavior during 2018 and 2019 Italian elections. Information, Communication & Society, 2020 | DOI — найти | ⚠️ | Окно координации (coordination interval), CooRnet |
| weber2021 | D. Weber, F. Neumann. Amplifying influence through coordinated behaviour in social networks. Social Network Analysis and Mining, 2021 | DOI — найти | ⚠️ | Временно́е окно, латентная сеть координации |
| nizzoli2021 | L. Nizzoli, S. Tardelli, M. Avvenuti, S. Cresci, M. Tesconi. Coordinated Behavior on Social Media in 2019 UK General Election. ICWSM 2021 | — | ⚠️ | Степень координации как континуум |
| chavoshi2016 | N. Chavoshi, H. Hamooni, A. Mueen. DeBot: Twitter Bot Detection via Warped Correlation. ICDM 2016 | — | ⚠️ | Синхронность активности как признак ботов |
| sharma2021amdn | K. Sharma et al. Identifying Coordinated Accounts on Social Media through Hidden Influence and Group Behaviours. KDD 2021 | — | ⚠️ | Бейзлайн AMDN-HAGE (из набора Iannucci) |
| nwala2023bloc | A. C. Nwala et al. A language framework for modeling social media account behavior (BLOC). EPJ Data Science, 2023 | — | ⚠️ | Бейзлайн BLOC |
| luceri2025tiktok | L. Luceri et al. Coordinated Inauthentic Behavior on TikTok: Challenges and Opportunities for Detection in a Video-First Ecosystem | arXiv:2505.10867 | 🔎+ | Перенос сетей сходства на платформы без ретвитов (синхронная загрузка) |

## LLM-эпоха: боты, кампании, симуляции

| Ключ | Источник | Идентификатор | Статус | Зачем |
|---|---|---|---|---|
| yang2024fox8 | K.-C. Yang, F. Menczer. Anatomy of an AI-powered malicious social botnet. J. Quantitative Description: Digital Media, 2024 | arXiv:2307.16336; DOI 10.51685/jqd.2024.icwsm.7 | ✅ (abs+HTML) | 1 140 ботов на ChatGPT; детекторы LLM-текста не отличают, координация видна; данные: Zenodo 10.5281/zenodo.8035290, открыто, CC BY 4.0, `fox8_23_dataset.ndjson.gz` 125 МБ (скачано 2026-11-17); README ✅ |
| ferrara2023 | E. Ferrara. Social bot detection in the age of ChatGPT: Challenges and opportunities. First Monday, 2023 | — | ⚠️ | Мотивация |
| qiao2025botsim | B. Qiao et al. BotSim: LLM-Powered Malicious Social Botnet Simulation. AAAI 2025 | arXiv:2412.13420; данные github.com/QQQQQQBY/BotSim (лицензия не указана) ✅ | 🔎+ | LLM-боты снижают качество классических детекторов; доп. данные |
| orlando2026emergent | G. M. Orlando, J. Ye et al. Emergent Coordinated Behaviors in Networked LLM Agents: Modeling the Strategic Dynamics of Information Operations. WWW 2026 | arXiv:2510.25003 | 🔎+ | LLM-агенты сами координируются: сети плотнее, усиление синхроннее → синхронность — устойчивый сигнал |
| olejnik2026iofactory | Ł. Olejnik et al. IO Factory: Simulating AI-Enabled Influence Campaigns at Scale. 2026 | arXiv:2608.10920 | 🔎+ | Масштабные симуляции AI-кампаний |
| cinus2025crossplatform | F. Cinus, M. Minici, L. Luceri, E. Ferrara. Exposing Cross-Platform Coordinated Inauthentic Activity in the Run-Up to the 2024 U.S. Election | arXiv:2410.22716 | 🔎+ | Скоординированные аккаунты в Telegram используют AI-контент чаще органики |
| ezzeddine2023 | F. Ezzeddine et al. Exposing influence campaigns in the age of LLMs: a behavioral-based AI approach to detecting state-sponsored trolls. EPJ Data Science 12:46, 2023 | arXiv:2210.08786 | 🔎+ | Поведение вместо языка как защита от LLM |
| openai2024io | OpenAI. Disrupting deceptive uses of AI by covert influence operations. Май 2024 | openai.com | 🔎+ | Реальные кампании Bad Grammar (комментарии ru/en в Telegram), Doppelganger |
| antonakaki2026 | D. Antonakaki, S. Ioannidis. Detecting Synthetic Political Narratives in Cross-Platform Social Media Discourse. 2026 (препринт) | arXiv:2605.21540 | 🔎+ | Ближайший «семантический» конкурент: индекс на уровне каналов; эталон в аннотации не описан |
| amplification2026 | Amplification to Synthesis: A Comparative Analysis of Cognitive Operations Before and After Generative AI. 2026 | arXiv:2605.13785 (авторы — проверить) | 🔎+ | Кластеры твитов по SBERT (порог 0,82) для перефразированных сообщений |
| seckin2025 | O. C. Seckin, M. Pote, A. Nwala, L. Yin, L. Luceri, A. Flammini, F. Menczer. Labeled Datasets for Research on Information Operations. ICWSM 2025 | arXiv:2411.10609; DOI 10.1609/icwsm.v19i1.35958; Zenodo 10.5281/zenodo.14141549 | ✅ (abs+HTML) | Данные: 26 кампаний + контроль; поля с текстом и временем. **Zenodo-запись (последняя версия 14189193) — доступ по заявке**: институциональная почта, 1 файл в день, правила IU о Foreign Entity of Concern; CC BY-NC-ND 4.0 (проверено 2026-11-17) |

## Детекция LLM-текста

| Ключ | Источник | Идентификатор | Статус | Зачем |
|---|---|---|---|---|
| krishna2023dipper | K. Krishna et al. Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense. NeurIPS 2023 | arXiv:2303.13408 | 🔎+ | Перефраз ломает индивидуальные детекторы; защита — поиск по сходству |
| shamardina2022ruatd | T. Shamardina et al. Findings of the RuATD Shared Task 2022 on Artificial Text Detection in Russian | arXiv:2206.01583 | 🔎+ | Русский язык, старые генераторы |
| multisocial2025 | MultiSocial: multilingual benchmark of machine-generated text detection of social-media texts | — | ⚠️ | Короткие тексты соцсетей |

## Россия, ВКонтакте

| Ключ | Источник | Идентификатор | Статус | Зачем |
|---|---|---|---|---|
| kolomeets2021friends | M. Kolomeets, A. Chechulin, I. Kotenko. Bot detection by friends graph in social networks. JoWUA 12(2), 2021 | isyou.info/jowua/papers/jowua-v12n2-6.pdf | 🔎+ | Боты ВКонтакте, графовый подход (СПб ФИЦ РАН) |
| kolomeets2023metrics | M. Kolomeets, A. Chechulin. Social bot metrics. Social Network Analysis and Mining, 2023 | данные github.com/guardeec/datasets (MKMETRIC2022) | 🔎+ | Рынок ботов ВКонтакте: цена, качество, скорость |
| kolomeets2024dissection | M. Kolomeets et al. From Detection to Dissection: Unraveling Bot Characteristics in the VKontakte Social Network. COMSNETS 2024 | — | 🔎+ | Honeypot-эксперименты с ботами VK |
| stukal2017 | D. Stukal, S. Sanovich, R. Bonneau, J. A. Tucker. Detecting Bots on Russian Political Twitter. Big Data, 2017 | — | 🔎+ | Боты в русскоязычном политическом сегменте |

## Методы (инструменты)

| Ключ | Источник | Статус | Зачем |
|---|---|---|---|
| traag2019leiden | V. A. Traag, L. Waltman, N. J. van Eck. From Louvain to Leiden: guaranteeing well-connected communities. Sci. Rep. 9, 2019 | ⚠️ | Выделение кампаний |
| bh1995 | Y. Benjamini, Y. Hochberg. Controlling the false discovery rate. JRSS B, 1995 | ⚠️ | Контроль ложных рёбер |
| e5_2024 | L. Wang et al. Multilingual E5 Text Embeddings: A Technical Report. 2024 (arXiv:2402.05672) | ⚠️ | Эмбеддинги (если выберем e5) |

## Свои работы

| Ключ | Источник | Статус | Зачем |
|---|---|---|---|
| frolov2026vkr | О. В. Фролов. ВКР «Разработка платформы мониторинга…», НИУ ВШЭ НН, 2026; код github.com/SamPetux/civic-monitor-nn | 📄 | Платформа, источник данных, ручное исключение ботов, перспектива «граф авторов» |
| rotanova2027burst | М. Б. Ротанова, О. В. Фролов, А. С. Шимичев. Zero-shot burst detection of civic activity… ElCon-CN 2027 (на рассмотрении) | 📄 | Корпус, детектор всплесков, хроника событий; ссылка — только если статья принята/опубликована |

## Найдено, но не использовать без первоисточника
- Google Research «S-CTS» (июнь 2026, сеть аккаунтов по SBERT) — известна только по пересказу в блоге, первичный
  текст не найден. Не цитировать.
