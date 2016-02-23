import matplotlib.pyplot as plt

perfs_525_5 = [(-33.41138830764555, 5.89552911786), (-33.41132330479954, 5.86016576521), (-33.41131592689711, 5.86346734291), (-33.411244459256935, 5.8993327081), (-33.411182260629715, 5.80064287624), (-33.411175461482486, 5.80065593227), (-33.411122143094666, 5.93335943476), (-33.41052072532104, 5.78816767193), (-33.40953472655828, 5.76671900847), (-33.408289226706856, 5.73858230012), (-33.234463271789345, 4.81028914322), (-33.1663914225156, 6.98496636871), (-33.04027187991658, 4.09907532472), (-33.008415221116046, 3.97935620066), (-32.97485545025183, 3.8486968732), (-32.926472543508716, 3.66856492822), (-32.8709755462958, 3.45868429901), (-32.83491393548434, 3.32116563839), (-32.81121149274013, 3.22575813279), (-32.783486687345906, 3.11591167306), (-32.70703265548308, 2.81361654903), (-32.6257391409826, 2.47975138441), (-32.5714854515892, 2.24579826628), (-32.558796163146724, 2.19164754404), (-32.4018250857535, 1.48895608607), (-32.32970461163653, 1.14863636494), (-32.20460548692487, 0.526492496155), (-32.11027403739196, 0.00517991612708), (-32.110252671036264, 0.00213403876798), (-32.10434601320493, 0.0), (-32.09341066525786, 9.29489403858), (-31.650523118388055, 11.9657001084), (-31.618542843053223, 25.0), (-31.58284648698906, 17.9748552824), (-31.569280565154187, 16.1246477195), (-31.55965478304961, 48.8402104922), (-31.557642250568776, 50.0)]

perfs_525_15 = [(-33.134361418265264, 4.81028914322), (-33.05555900454694, 4.09907532472), (-33.019678895104505, 3.97935620066), (-32.98365129500627, 3.8486968732), (-32.93346804205811, 3.66856492822), (-32.87536731811073, 3.45868429901), (-32.83869967666741, 3.32116563839), (-32.81377371633453, 3.22575813279), (-32.78582711207762, 3.11591167306), (-32.70848146757618, 2.81361654903), (-32.62628285721852, 2.47975138441), (-32.57178915722108, 2.24579826628), (-32.558959639258234, 2.19164754404), (-32.401881964813, 1.48895608607), (-32.32970219118212, 1.14863636494), (-32.20460548692487, 0.526492496155), (-32.11027403739196, 0.00517991612708), (-32.110252671036264, 0.00213403876798), (-32.10434601320493, 0.0), (-31.21545780784069, 5.73858230012), (-31.15188139116149, 5.76671900847), (-31.103978492314614, 5.78816767193), (-31.078154610840652, 5.80064287624), (-31.07811596853352, 5.80065593227), (-30.961674008888, 5.86016576521), (-30.95547730640542, 5.86346734291), (-30.897058404054714, 5.89552911786), (-30.89021154127514, 5.8993327081), (-30.82782206177496, 5.93335943476), (-30.342236404016624, 17.9748552824), (-30.336502184947157, 16.1246477195), (-30.28149193966931, 25.0), (-30.209757997241137, 9.29489403858), (-30.12676284651617, 48.8402104922), (-30.126296489181495, 50.0), (-29.954393670580252, 11.9657001084), (-29.392058211661702, 6.98496636871)]

perfs_525_25 = [(-33.055558876160504, 4.09907532472), (-33.01973704458892, 3.97935620066), (-32.983639274047896, 3.8486968732), (-32.93347241678388, 3.66856492822), (-32.87536937099375, 3.45868429901), (-32.838702972567404, 3.32116563839), (-32.81377371633453, 3.22575813279), (-32.78582777099431, 3.11591167306), (-32.70848146757618, 2.81361654903), (-32.62628285721852, 2.47975138441), (-32.61312664038021, 4.81028914322), (-32.57178915722108, 2.24579826628), (-32.558959639258234, 2.19164754404), (-32.401881964813, 1.48895608607), (-32.32970219118212, 1.14863636494), (-32.20460548692487, 0.526492496155), (-32.11027403739196, 0.00517991612708), (-32.110252671036264, 0.00213403876798), (-32.10434601320493, 0.0), (-30.215940255942208, 5.73858230012), (-30.166572438107192, 5.76671900847), (-30.129149030054638, 5.78816767193), (-30.106763420369354, 5.80064287624), (-30.106752423965027, 5.80065593227), (-29.98537589361003, 5.86016576521), (-29.97669854055003, 5.86346734291), (-29.898703389883536, 5.89552911786), (-29.888956730966918, 5.8993327081), (-29.880848001918686, 9.29489403858), (-29.79781676441866, 5.93335943476), (-29.75080389863451, 17.9748552824), (-29.694491043379237, 25.0), (-29.689092033370656, 16.1246477195), (-29.424035853444593, 11.9657001084), (-29.40014937526262, 50.0), (-29.397830377816202, 48.8402104922), (-28.42138645159279, 6.98496636871)]

scores5 = []
params5 = []

scores15 = []
params15 = []

scores25 = []
params25 = []

for score, param in perfs_525_5:
    scores5.append(-score)
    params5.append(param)

for score, param in perfs_525_15:
    scores15.append(-score)
    params15.append(param)

for score, param in perfs_525_25:
    scores25.append(-score)
    params25.append(param)


plt.plot(params5, scores5, 'ro', label= "5 iterations")
plt.plot(params15, scores15, 'bo', label = "15 iterations")
plt.plot(params25, scores25, 'go', label = "25 iterations")
plt.legend()
plt.show()
