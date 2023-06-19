
subcorticalROIs = [129 123 258 130 100:119 120 217  125 259 122 260 131 133 251 266 267 268 132 265 126 262 128 263 127 264 88 224 124 261 121 257];
cereb =[ 245 248 244 265 243 238 250 240 242 241 247 253 254 249 243 250 237 236 255 246 252 239 256 ]

tokeep = [subcorticalROIs, cereb]

v = niftiread('/Users/emilyolafson/GIT/ENIGMA/testfig/shen268_MNI1mm_dil1.nii');

hdr = niftiinfo('/Users/emilyolafson/GIT/ENIGMA/testfig/shen268_MNI1mm_dil1.nii');
    
shencopy = zeros(182,218,182);
for i = 1:length(toremove)
    shenBool = v == tokeep(i);
    shencopy(shenBool) =tokeep(i);
end
shencopy=single(shencopy);
niftiwrite(shencopy, '/Users/emilyolafson/GIT/ENIGMA/testfig/shen268_MNI1mm_dil1_subcort.nii', hdr);