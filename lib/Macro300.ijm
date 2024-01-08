for(i=10; i<55; i++){
open("D:/Universita/Tesi/Dataset_robbi/DBT_card_original/16-00"+i+".tif");
run("Size...", "width=64 height=64 depth=1 average interpolation=Bilinear");
saveAs("PNG", "D:/Universita/Tesi/Dataset_robbi/DBT_card_64/"+i+".tif");
close();
}
