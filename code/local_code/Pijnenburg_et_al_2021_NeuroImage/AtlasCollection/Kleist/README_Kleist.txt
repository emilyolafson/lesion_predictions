Version 1; 15-01-2021; Free to use under a Creative Commons Attribution-NonCommercial-Share A like 4.0 International License.

Package content:
README_Kleist							Data version, licensing information and list of brain regions, including for each region: region number, digital atlas label, the corresponding original region and allocated function in English and German
'label' folder							Containing individual manually drawn labels, used in atlas digitisation
lh.atlas.gcs & rh.atlas.gcs					Surface-based atlas
lh.fsaverage.Kleist.label.gii rh.fsaverage.Kleist.label.gii 	Surface-based atlas, applied to fsaverage and converted to .gii surface mesh using mris_convert 
Kleist.nii.gz							Colin27 volume-based atlas
Kleist_NCBI152.nii.gz						NCBI152 volume-based atlas
lh.colortable.txt & rh.colortable.txt				Colortable files for the surface-based atlas
KleistColorLUT.txt						Lookup table, including ROI number and color code for each region

NOTES: 
Some Kleist regions are subdivided into several segments. In the digital labels, these segments are specified by suffixes "_1", "_2" and "_3".


List of Brain Regions
#	Digital label			Original Region				Function (translated from German to English)						Function (German)
1	K1_3a_3b			1, 3a and 3b				Touch sensation, Pain sensation, Temperature sensation					Berührungsempfindung, Schmerzempfindung, Temperaturempfindung		
2	K2				2					Kinesthetic sensation									Kinästethische Empfindung
3	K4				4					Individual movements									Einzelbewegungen
4	K5				5					No functional allocation								-	
5	K6aa				6a alpha				Dexterity, Power perception, Tone and sound formation					Fertigkeiten, Kraftempfindung, Ton und Laut Bildung
6	K6ab_1				6a beta (upper segment)			Trunk turning										Rumpfwendungen
7	K6ab_2				6a beta (lower segment)			Head turning										Kopfwendungen
8	K7				7					Action (sensory) leg, trunk								Handeln (sensorisch) Bein, Rumpf
9	K8_1				8 (upper segment)			Falling and pointing reactions								Fall- und Zeige Reaktionen
10	K8_2				8 (lower segment)			Ocular turning										Augenwendungen
11	K9				9					Initiative, Awareness of effort and power						Antrieb, Anstrengungs- und Kraftgefühle
12	K10				10					Motor skill										Motorische Handlungsfolgen
13	K11_12				11 and 12				Personal and social ego									Selbst- und Gemeinschafts Ich
14	K13				13					No functional allocation								- 
15	K16				16					No functional allocation								- 
16	K17_1				17 (lateral segment)			Vision: brightness, colors, forms, movements						Sehen: Helligkeiten, Farben, Formen, Bewegungssehen
17	K17_2				17 (medial upper segment)		Visual field, lower quadrant								Gesichtsfeld, unterer Quadrant
18	K17_3				17 (medial lower segment)		Visual field, upper quadrant								Gesichtsfeld, oberer Quadrant
19	K18_1				18 (lateral segment)			Sense of place, Eye movements, Optic awareness						Ortssinn, Blickbewegungen, Optische aufmerksamkeit
20	K18_2				18 (medial upper segment)		Conjugate downward eye movements							Blickbewegungen nach unten
21	K18_3				18 (medial lower segment)		Conjugate upward eye movements								Blickbewegungen nach oben
22	K19_1				19 (lateral segment)			Calculation, Recognition of numbers, Reading, Visual thinking, Visual recognition	Rechnen, Zahlenerkennen, Lesen, Optische Gedanken, Optische Dingerkennen
23	K19_2				19 (medial upper segment)		Place memory										Orts gedächtnis
24	K19_3				19 (medial lower segment)		Color and object recognition								Farben- Dingerkennen
25	K20				20					Appreciation of sounds and music							Sinnverständnis für Geräusche und Musik
26	K21				21					Hearing movements, Acoustic awareness							Horchbewegungen, Akustische Aufmerksamkeit
27	K22a_1				22a (upper segment)			Noise understanding									Geräuschfolgen
28	K22a_2				22a (lower segment)			Melody understanding									Tonfolgen (Melodieverständnis)
29	K22b_1				22b (lower segment)			Word understanding									Lautfolgen (Wortverständnis)
30	K22b_2				22b (upper segment)			Sentence understanding									Satzverständnis
31	K23_24_26_29_30_31_32_33	23, 24, 26, 29, 30, 31, 32 and 33	Corporeal ego (personal experience/awareness)						Körper Ich (Eigen Erleben)
32	K25				25					No functional allocation								-
33	K27				27					No functional allocation								-
34	K28_34				28 and 34				Olfactory recognition									Geruchserkennen
35	K35_36				35 and 36				Representative olfactory movements							Gegenständliche Geruchsbewegungen
36	K37				37					Name understanding									Namen verständnis
37	K38				38					No functional allocation								-
38	K39_1				39 (upper segment)			Constructive action (sensory)								Konstruktives Handeln (sensorisch)
39	K39_40				39 and 40 (shared segment)		Body image, Right-Left orientation							Körper Tastbild, Rechts-Links Oriëntierung
40	K40_1				40 (upper segment)			Individual consecutive action								Einzelhandlung, Handlungsfolgen
41	K40_2				40 (posterior segment)			Recognition by touch									Ding tasterkennen
42	K40_3				40 (anterior segment)			Face activity (sensory)									Gesicht handeln (sensorisch)
43	K41_42_52			41, 42 and 52				Sound/melody sensation, Noise sensation, Loudness sensation				Tonempfindungen, Geräuschempfindungen, Lautempfindungen
44	K43				43					Taste											Geschmack
45	K44a				44a					Melody-, word formation									Melodie-, Wortbildung
46	K44b				44b					Name speaking (spontaneous) 								Namen (Spontan) Sprechen
47	K45				45					Sentence speaking									Satzsprechen
48	K46				46					Constructive thinking									Tätige Gedanken
49	K47				47					Sentiment/Attitude, Mood-affected actions, Perseverance					Gesinnungen, Gesinnungmäßige Handlungen, Ausdauer