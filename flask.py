from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
app=Flask("_name_")
m=pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("nara.html")

@app.route("/Yield")
def Yield():
    return render_template("Yield prediction.html")
@app.route("/data")
def data():
    return render_template("data.html")
@app.route("/blog")
def blog():
    return render_template("blog.html")
@app.route("/predict",methods=["POST","GET"])
def predict():
    d=request.form
    districts=['District_Adilabad', 'District_Bhadradri Kothagudem','District_Jagtial', 'District_Jangoan', 'District_Jayashankar Bhoopalpally','District_Jogulamba Gadwal', 'District_Kamareddy', 'District_Karimnagar','District_Khammam', 'District_Komaram bheem asifabad','District_Mahabubabad', 'District_Mahabubnagar', 'District_Mancherial','District_Medak', 'District_Medchal-Malkajgiri', 'District_Mulug','District_Nagarkurnool', 'District_Nalgonda', 'District_Narayanpet','District_Nirmal', 'District_Nizamabad', 'District_Peddapalli','District_Rajanna Sircilla', 'District_Rangareddy', 'District_Sangareddy','District_Siddipet', 'District_Suryapet', 'District_Vikarabad','District_Wanaparthy', 'District_Warangal', 'District_Hanumakonda','District_Yadadri Bhuvanagiri']
    dis=[0]*32
    dis[districts.index(d["district"])]=1
    season=[0,0]
    if(d["season"]=="Season_Kharif"):
        season[0]=1
    else:
        season[1]=1
    crops=['Crop_Groundnut', 'Crop_Maize','Crop_Moong(Green Gram)', 'Crop_Rice', 'Crop_cotton(lint)']
    crop=[0]*5
    #crop[crops.index(d['crop'])]=1
    for i in range(len(crops)):
        if crops[i]==d["crop"]:
            crop[i]=1
    l=list(d.values())
    r=l[3:15]
    l1=r+dis+season+crop
    a=np.array([l1])
    a = np.array(a, dtype=float)
    print(a)
    out = m.predict(a)

    if d["crop"] == 'cotton(lint)':
        return render_template("Yield prediction.html", out=f" {round(out[0],5)} Bales/Hectare")
    else:
        return render_template("Yield prediction.html", out=f" {round(out[0],5)} Tonnes/Hectare")


app.run(host="0.0.0.0", port=5000)