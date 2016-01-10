
var fs  = require('fs');
var f1 = fs.readFileSync("/Users/grahamgibson/Downloads/pdb2mx7.ent");
var sys = require('sys');
var fileString = f1.toString();
var patt = /\s+/;
var form = fileString.split(patt);
var input = [];
var output = [];

for(var i  =0; i < form.length; i++){
	if(form[i] === "SEQRES"){
		//first seqres
		//find next one
		for (var j = i+4; j < i+ 16; j++){
			input.push(form[j]);
		}







	}
}



var k2 = 0;
for(var i2  =0; i2 < form.length; i2++){
	if(form[i2] === "ATOM"){
		//find next one
		 for (var j2 = i2+6; j2 < i2+6 + 3; j2++){
			output.push(form[j2]);
		 }






	}
}
input = input.toString().split("HELIX")[0];
output = output.toString();


fs.appendFile('/Users/grahamgibson/Downloads/output.txt', output, function (err) {

});

var inputArray = input.split(",");
//we now need to map the input to amino acid pdb files
var jsonMap = {
        "ALA":       "alanine",
        "ARG" :      "arginine",
        "ASN"     :      "asparagine",
        "ASP"     :       "aspartate",
        "CYS"     :       "cysteine",
        "GLN"       :     "glutamine",
        "GLU"    :      "glutamate",
        "GLY"     :       "glycine",
        "HIS"     :       "histidine",
        "ILE"     :       "isoleucine",
        "LEU"     :       "leucine",
        "LYS"     :       "lysine",
        "MET"     :       "methionine",
        "PHE"     :       "phenylalanine",
        "PRO"     :       "proline",
        "SER"     :       "serine",
        "THR"     :       "threonine",
        "TRP"     :       "tryptophan",
        "TYR"     :       "tyrosine",
        "VAL"     :       "valine",
        "ASX"     :       "asparagine"
    };


var aminoAcidPDB = [];
for(var k1 = 0; k1 < inputArray.length; k1++){
	if(inputArray[k1]){
		var fi = fs.readFileSync("/Users/grahamgibson/Downloads/" + jsonMap[inputArray[k1]] + ".pdb").toString();
		aminoAcidPDB.push(fi.split(patt));
	}
}



var aminoAcidCoordinates = [];
var formattedOutput = [];
for(var j1 = 0; j1 < aminoAcidPDB.length; j1++){
	var tmp = [];
	for(var i2  =0; i2 < aminoAcidPDB[j1].length; i2++){
		if(aminoAcidPDB[j1][i2] === "ATOM"){
			//first seqres
			//find next one
			 for (var j2 = i2+4; j2 < i2+4 + 3; j2++){
				tmp.push(aminoAcidPDB[j1][j2]);
			 }

		}
	}
	formattedOutput.push(tmp);
}

//add to the x coordinates
var finalFormatOutput = [];
var newIter = 1;
for(var casey = 0; casey < formattedOutput.length; casey++){
	for (var casey2 = 0; casey2 < formattedOutput[casey].length; casey2++ ){
		if(casey2 % 3 === 0){
			finalFormatOutput.push(formattedOutput[casey][casey2] + 10*newIter);

		} else {
						finalFormatOutput.push(formattedOutput[casey][casey2]);

		}
	}
	newIter ++;
}



fs.appendFile('/Users/grahamgibson/Downloads/input.txt', finalFormatOutput, function (err) {

});

console.log(finalFormatOutput.length);
