# Baseline Results Report

> **번호 체계 안내:** 이 문서의 EXP-XXX는 2026-07-04 노트북 작업 시점 번호이며, main의 `docs/EXPERIMENT_LOG.md` 최신 체계(EXP-002~011, 2026-07-06 이후 정리)와 다르다. 이 문서의 **EXP-006(SillokBERT CPU fine-tune)은 main 기준 EXP-011**에 해당한다(`docs/DECISIONS.md` [009] 참고). 그 외 번호(EXP-002~005/00Xb, GPU EXP-007)는 main과 1:1 대응되지 않을 수 있으니 새 실험 번호 부여 전 교차 확인할 것.


- Generated: 2026-07-04
- Device: **CPU**
- Eval split: `test`
- Primary comparison: **n=2000 subsample, seed=42** (EXP-002b through EXP-006)

## Data integrity (pre-FT gate)

- Mask dataset rows: 66537
- Inscription split integrity: train∩test=0, train∩valid=0, valid∩test=0 (PASS)
- SillokBERT UNK labels on full test: 233 / 9719 (2.4%)

## Overall Exact@k (test subsample n=2000)

| EXP | Model | n | Exact@1 | Exact@5 | Exact@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| EXP-002b | frequency | 2000 | 0.0230 | 0.0675 | 0.1075 |
| EXP-003b | ngram | 2000 | 0.1570 | 0.2865 | 0.3475 |
| EXP-004 | bert-base-chinese | 2000 | 0.1975 | 0.3555 | 0.4245 |
| EXP-005 | ddokbaro/SillokBert | 2000 | 0.2365 | 0.4260 | 0.5090 |
| EXP-006 | outputs/finetune/sillokbert_gsm_baseline/final | 2000 | 0.3130 | 0.5045 | 0.5775 |

## Full test reference (n=9719, non-DL only)

| EXP | Model | n | Exact@1 | Exact@5 | Exact@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| EXP-002 | frequency | 9719 | 0.0225 | 0.0660 | 0.1033 |
| EXP-003 | ngram | 9719 | 0.1448 | 0.2807 | 0.3467 |

## Era breakdown (EXP-006 FT)

| Era | n | Exact@1 | Exact@5 | Exact@10 |
| --- | ---: | ---: | ---: | ---: |
| 고구려 | 4 | 0.2500 | 0.2500 | 0.2500 |
| 고려 | 221 | 0.2851 | 0.4751 | 0.5430 |
| 낙랑 | 11 | 0.1818 | 0.3636 | 0.3636 |
| 대한제국 | 75 | 0.2000 | 0.3733 | 0.4267 |
| 백제 | 2 | 0.0000 | 1.0000 | 1.0000 |
| 신라 | 2 | 0.0000 | 0.0000 | 0.0000 |
| 일제강점기 | 112 | 0.2411 | 0.4911 | 0.5625 |
| 조선 | 1546 | 0.3325 | 0.5201 | 0.5957 |
| 통일신라 | 27 | 0.1481 | 0.3704 | 0.4444 |

## Failure cases (FT model, Exact@10 miss, up to 20)

| sample_id | label | top5 | input (truncated) |
| --- | --- | --- | --- |
| mask_00036368 | 異 | 文;遺;仁;之;風 | 粹之姿究厥底蘊措諸聲施翼翼漢都神明攸基颺言獻圖奠鼎居師弼亮四世致」理雍熙王曰相予汝翼汝爲懿彼壽俊憲其令儀端委廟朝立經陳䂓禮俗廼惇[MASK]敎斯麾人誦屋傘官補園 |
| mask_00017580 | 公 | ,;。;也;始;誕 | 東甘泉人也國初有諱曰龜爲政丞高祖漢英內部令曾祖淑器寶城監務祖孫貫安東判官連三世皆由正科考諱續命典涓司直長妣秦氏江順府少尹有經之女[MASK]生於天順丁丑十月成化 |
| mask_00037435 | 幕 | 至;。;,;後;自 | 者以要名誤事為罪名公編配丹陽郡既蒙放不欲涉迹於世乃曰此地深僻去人甚逺遂仍居不出為終焉計越四年庚辰以親意勉應朝命由禮曺正郞出佐湖西[MASK]壬午再有校理之命㦯辭 |
| mask_00052908 | 於 | 日;後;夜;歸;公 | 恙不廢爲官無小大一意奉公不間雜以一毫私情非惟公不以關莭及人人亦不敢以關莭浼公公退靜坐一室左右圖書以終日唯復蒔養松竹相與瀟洒而已其[MASK]還徃馳遂断如也不以家 |
| mask_00021751 | 死 | 名;人;子;文;事 | 疇東九庶曾孫也主公祀者東毅之子泰鎭也系之以銘曰昔歲辛壬群凶盜權文武駢首死也光鮮公膽如斗在獄彌激懼死不亟張目叱賊將則盡矣相豈有書其[MASK]誰爲爲我國儲度遼恥也 |
| mask_00015256 | 門 | 中;列;會;域;內 | 瑞異蟲蛇遁其毒形松暗雲深夏涼冬燠斯三韓勝地也禪師擁錫來遊乃有縣車之意爰開敎」化之場用納資禀之客漸頓雲集於四禪之室賢愚景附於八定之[MASK]縱有波旬之儻梵志之徒 |
| mask_00015293 | 四 | 以;而;之;不;打 | 本空去來常寂不視生滅濟度凡」迷前諸未度忽失前緣已得後度湏達理者以爲報盡形謝而痛惜哉於焉輟斤絕絃也終前三往所」㞐山北而令伐杉樹大可[MASK]圍曰有人死則將此作函 |
| mask_00053446 | 弼 | 佐;武;海;漢;文 | 贈嘉善大夫戶曹叅判兼同知義禁」府事五衛都捴府副捴管行通政」大夫敦寧府都正鄭公采和之墓」公諱采來和字後仙東萊人中廟朝名相謚文翼諱光[MASK]五代孫曾祖左相號林塘 |
| mask_00050854 | 工 | 嘗;之;曾;生;常 | 天之生公豈偶然也哉惜乎飛不盡翰歛退林樊以終其年其視誇功衒能貪進不知止者可謂賢於人一等而垂誡後承未嘗不出自這裡亦豈非可以知公者歟公[MASK]於詩嘗有絕影島一詠與 |
| mask_00042259 | 玄 | 孫;未;祖;子;考 | 四男二女男東九東培東洛東龍女洪萬基李宇鎬景源一男二女男東植女金永壽尹仁善洛源二男五女男東烈東輝女金義一南軍鉉成殷慶鄭秉錫餘未婚曾[MASK]以下不錄嗚乎公志于學 |
| mask_00049417 | 番 | 軍;士;衛;將;諸 | 始作北門廣設營舍未竟布置而罷越癸未夏余承命考史于閣仍見城內僅有守直番卒若于僧軍數指亦踐更往來鉢囊無所庇倉積儲胥一無素且聞丁丑之變[MASK]卒皆散守僧頗解事私搬 |
| mask_00053453 | 佐 | 男;公;生;女;子 | 日卒一男載道司宰僉正參保社原從贈公參判三女通」德尹之憲進士洪湙士人李秀廷僉正三男受先重先進士潤先正郞尹三男爀縣監次煜煥女縣監李箕[MASK]洪一」女李樀受先一男 |
| mask_00023692 | 君 | 州;陽;原;川;安 | 於經亂之後覽數公殉義之地爲之俛仰咨嗟旣又詢邑之父老得數公殉義始末甚詳謂樹民彝在乎是折奸萠在乎是就城中建三忠祠竝享李南二公而配以洪[MASK]士人朴之垕衞將崔泰敬 |
| mask_00041473 | 寃 | 冤;辨;公;狀;之 | 君行判敦寧府事李忠正公墓誌銘第六共十一奇以禽方張之賊哉假使有功棄軍之罪固自如也廷弼竟以此抵罪其後臺臣趙侹又論其罪廷弼子寅徽擊鼓訟[MASK]以爲監司之狀由於鄭暘 |
| mask_00004732 | 評 | 氏;公;範;銑;欽 | 宜人恩津金氏之墓參奉安東權公諱[MASK]之墓 |
| mask_00049418 | 守 | 在;亡;,;匿;其 | 廣設營舍未竟布置而罷越癸未夏余承命考史于閣仍見城內僅有守直番卒若于僧軍數指亦踐更往來鉢囊無所庇倉積儲胥一無素且聞丁丑之變番卒皆散[MASK]僧頗解事私搬史櫃藏之 |
| mask_00020392 | 天 | 然;彼;今;夫;而 | 知耻如公之不汙其君不私其身蹈白刄而無悔使人知仁義之可慕者可不謂」豪傑之士乎嗚呼夫以公所有而得時行志其笙鏞一世以鳴國家之盛者豈異哉[MASK]旣生如此人而又生子光 |
| mask_00056855 | 淑 | 任;新;凝;濟;貞 | 余甞遊二先生門者託以銘文雖不敢當而亦有所不敢辭者謹按公諱澈字季涵迎日人遠祖均之麗朝平章事本朝有洪諡恭簡子淵兵曹判書是公高祖曾祖自[MASK]郡守祖潙參奉考惟沈判 |
| mask_00064222 | 達 | 憲;公;翊;顯;植 | 牧使有二男志式㕘奉志翼縣令余有二男一女琪玗今爲綂制使女壻崔台鎭判書有三男三女時黙進士持黙女壻李憲鎮餘㓜啓遠男達海同知啓健男女㓜李[MASK]之繼男夏禎宗海娶縣令 |
| mask_00023046 | 則 | 而;時;後;事;以 | 不念老舅之情耶即掩淚柔聲而」止自此以後血淚雖濕於枕衾愁色不見於老舅徃在庚辰舅有長瘧累月不差醫藥無效問於高師則某崇在某方當夜」致祭[MASK]見效云故即待其夜適有 |

## Limitations

- **Split:** inscription-level split; cross-split inscription overlap verified as 0.
- **Tokenizer:** SillokBERT UNK label ratio on test ≈ 2.4%. Training skips UNK labels; eval includes all subsampled rows.
- **Context:** MLM eval uses ±64 char window; sequences >512 tokens skipped in FT.
- **FT scale:** CPU run used train subsample max_train=8000 (not full 46,139).
- **Eval scale:** Primary table uses n=2000 subsample (seed=42) for fair DL comparison.
- **SillokBERT advantage:** may partly reflect Joseon annals pretraining overlap with formulaic epigraphic phrases—not fully separated from generalization in this baseline.
- **Mask type:** single-character artificial masks only; consecutive defects (□□□) not covered.
- **Real defects:** qualitative demo only (`outputs/predictions/real_defect_demo.csv`).

## Output paths

- `outputs/evaluation/evaluation_summary.csv`
- `outputs/evaluation/evaluation_summary_canonical.csv`
- `outputs/predictions/exp002b_frequency_sub2k_predictions.csv`
- `outputs/predictions/exp003b_ngram_sub2k_predictions.csv`
- `outputs/predictions/exp004_bert_zs_predictions.csv`
- `outputs/predictions/exp005_sillok_zs_predictions.csv`
- `outputs/predictions/exp006_sillok_ft_predictions.csv`
- `outputs/finetune/sillokbert_gsm_baseline/final/`
- `data/model_inputs/real_defect_inference.csv`
- `outputs/predictions/real_defect_demo.csv`
- `work/step1_verification.json`
