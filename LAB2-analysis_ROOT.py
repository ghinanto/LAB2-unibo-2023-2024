from ROOT import TCanvas, TF1, TGraphErrors, TLatex, TText, TLegend, TH1D, Double_t, TH1F
from ROOT import gROOT, gStyle, gPad
import ROOT
import numpy as np
import pandas as pd
from array import array
import sys
import matplotlib.pyplot as plt
#import seaborn as sns
import statistics as st
#from lmfit.models import LinearModel
import os
#gROOT.SetStyle("CMS")

def get_delay_from_filename(filen):
    return int(os.path.basename(filen).removeprefix("Calibrazione_").split("_")[0])

def get_mpv(data_df, column_name):
    dups = data_df.pivot_table(index = [column_name], aggfunc ='size')
    dups_df = pd.DataFrame({"value":dups.index, "counts": dups.values})
    #print(dups_df)
    mpv = np.mean(data_df[column_name].values)
    mpv_stddev = st.stdev(data_df[column_name].apply(lambda x: x if type(x) is float else float(x))) #[int(str(ticks), 16) for ticks in data_df[column_name].values])
    return mpv, mpv_stddev

def fix_ticks_measure(ticks_mpv, ticks_mpv_stddev):
    if (ticks_mpv_stddev == 0.):
      ticks_mpv_stddev = 0.25
      ticks_mpv += 0.5
    return ticks_mpv, ticks_mpv_stddev

def get_calibration_parameters(calibration_data_files):
    """
    This function accept calibration data files, each file containing data for a particular delay value
    with a name of the format 'Calibrazione_108_R3.txt'. If your file name scheme is different just modify
    the function 'get_delay_from_filename' that is used to get the delay value.
    The data points to fit with a linear model are chosen as the most probabe values (the most frequent ones) of the dataset and the variance of the dataset used as error. 
    The output is slope and intercept of linear fits for each scintillator.
    """
    # FIRST CELL
    calibration_dict = {"P1": pd.DataFrame([]), "P2": pd.DataFrame([]), "P3": pd.DataFrame([])}
    calibration_dict_list = {"P1": [], "P2": [], "P3": []}

    for filen in calibration_data_files:
        #print(filen)
        delay_ns_from_filename = get_delay_from_filename(filen)

        calibration_data_df = pd.read_table(filen, sep=' ', names=['None', 'Sample', 'Pulse_Timing_micros', 'delay_ticks_P1', 'delay_ticks_P2', 'delay_ticks_P3'])
        calibration_data_df["delay_ticks_P1"] = calibration_data_df["delay_ticks_P1"].apply(lambda x: int(str(x), 16))
        calibration_data_df["delay_ticks_P2"] = calibration_data_df["delay_ticks_P2"].apply(lambda x: int(str(x), 16))
        calibration_data_df["delay_ticks_P3"] = calibration_data_df["delay_ticks_P3"].apply(lambda x: int(str(x), 16))

        #print(calibration_data_df)

        ticks_mpv, ticks_mpv_stddev = get_mpv(calibration_data_df, "delay_ticks_P1")
        ticks_mpv, ticks_mpv_stddev = fix_ticks_measure(ticks_mpv, ticks_mpv_stddev)
        calibration_dict_list["P1"].append([delay_ns_from_filename, ticks_mpv, ticks_mpv_stddev])#delay_ns_mpv, delay_ns_mpv_stddev, ticks_mpv, ticks_mpv_stddev])

        ticks_mpv, ticks_mpv_stddev = get_mpv(calibration_data_df, "delay_ticks_P2")
        ticks_mpv, ticks_mpv_stddev = fix_ticks_measure(ticks_mpv, ticks_mpv_stddev)
        calibration_dict_list["P2"].append([delay_ns_from_filename, ticks_mpv, ticks_mpv_stddev])#delay_ns_mpv, delay_ns_mpv_stddev, ticks_mpv, ticks_mpv_stddev])

        ticks_mpv, ticks_mpv_stddev = get_mpv(calibration_data_df, "delay_ticks_P3")
        ticks_mpv, ticks_mpv_stddev = fix_ticks_measure(ticks_mpv, ticks_mpv_stddev)
        calibration_dict_list["P3"].append([delay_ns_from_filename, ticks_mpv, ticks_mpv_stddev])#delay_ns_mpv, delay_ns_mpv_stddev, ticks_mpv, ticks_mpv_stddev])

    delay_ns_errors = [24e-3, 95e-3, 264e-3, 379e-3, 230e-3]
    for key in calibration_dict:
        calibration_dict[key] = pd.DataFrame(calibration_dict_list[key], columns=["delay_ns", "delay_ticks_mpv", "delay_ticks_mpv_stddev"])
        calibration_dict[key]["tick_duration_ns"] = calibration_dict[key]["delay_ns"] / calibration_dict[key]["delay_ticks_mpv"]
        calibration_dict[key]["delay_ns_stddev"] = delay_ns_errors

        #print("\nScintillator " + key + ":")
        #print(calibration_dict[key])


    # SECOND CELL # FIT
    calibration_parameters = {"P1":{}, "P2":{}, "P3":{}}

    #canvas = {"P1":TCanvas( 'c1', 'c1', 700, 500), "P2":TCanvas( 'c1', 'c1', 700, 500), "P3":TCanvas( 'c1', 'c1', 700, 500)}
    #canvas_list = [TCanvas( 'c1', 'c1'), TCanvas( 'c2', 'c2'), TCanvas( 'c3', 'c3')]
    #i = 0
    canvas_list = []
    for key in calibration_dict:
        canvas = TCanvas( 'c' + key, 'c' + key)
        canvas.SetCanvasSize(700,500)
        #print("\nScintillator " + key + ":")
        #calibration_dict[key].plot("delay_ticks_mpv", "delay_ns", kind="scatter")

        canvas.SetGrid()

        x  = array( 'f', calibration_dict[key]["delay_ticks_mpv"].values )
        ex = array( 'f', calibration_dict[key]["delay_ticks_mpv_stddev"].values )
        y  = array( 'f', calibration_dict[key]["delay_ns"].values )
        ey = array( 'f', calibration_dict[key]["delay_ns_stddev"].values)

        gr = TGraphErrors( len(calibration_dict[key]["delay_ticks_mpv"].values), x, y, ex, ey , )
        gr.SetMarkerStyle(20)
        gr.SetMarkerSize(1.3)
        gr.GetXaxis().SetTitle("FPGA Clock Ticks")
        gr.GetYaxis().SetTitle("Time Difference (ns)")
        #gr.SetMarkerColor( 4 )
        #gr.SetMarkerStyle( 21 )
        funct = TF1("funct", "[0] * x + [1]", 0, 6)
        funct.SetParName(0, "m")
        funct.SetParName(1, "q")
        gr.Fit("funct")
        gr.SetTitle( 'Calibration of ' + key )
        funct.SetTitle('Calibration of ' + key)
        gr.Draw( 'ALP' )
        funct.Draw("same")

        canvas.SetLeftMargin(0.14)

        gStyle.SetOptFit("e")
        gStyle.SetOptFit(1)
        #legend = canvas.BuildLegend(0.15, 0.7, 0.4, 0.9)
        #legend = TLegend(0.15, 0.7, 0.4, 0.9)

        ps = gr.GetListOfFunctions().FindObject("stats")
        ps.SetX1NDC(0.2)
        ps.SetX2NDC(0.6)
        ps.SetY1NDC(ps.GetY1NDC()-0.05)
        ps.SetY2NDC(ps.GetY2NDC()-0.05)

        #lines = ps.GetListOfLines()
        #TText line_p0 = ps.GetLineWith("p0")

        canvas.Modified()
        canvas.Update()


        canvas.Update()
        #canvas.Draw()
        canvas_list.append(canvas)
        #canvas.Draw()

        #print(f"a = {funct.GetParameter(0):.6f} +/- {funct.GetParError(0):.6f}")
        #print(f"b = {funct.GetParameter(1):.6f} +/- {funct.GetParError(1):.6f}")
        calibration_parameters[key] = {"slope": funct.GetParameter(0), "slope_err": funct.GetParError(0), "intercept": funct.GetParameter(1), "intercept_err": funct.GetParError(1)}


        canvas.Print("Calibration_" + key + ".pdf")

    return calibration_parameters

def ticks_to_ns(tdc_ticks, slope_tdc, offset_tdc):
  return slope_tdc*tdc_ticks + offset_tdc

def get_datasets(data_file, datasets, calibration_parameters):

    datasets_dict = {}
    pmts_all = ["P1", "P2", "P3"]

    #TDC (time delay counts)

    # Read the files, append to list only if at least one plane has a valid TDC stop
    data_df = pd.read_table(data_file, sep=' ', names=['None', 'Sample', 'Pulse_Timing_micros', 'delay_ticks_P1', 'delay_ticks_P2', 'delay_ticks_P3'])

    # convert hexadecimal values to decimal ones (e.g. 'fff' becomes 4095)
    for col in ['delay_ticks_P1', 'delay_ticks_P2', 'delay_ticks_P3']:
        data_df[col] = data_df[col].apply(lambda x: int(x, 16))
    datasets_dict["all_data"] = data_df

    #remove empty events
    tdc_nonempty = [tdc for tdc in zip(data_df['delay_ticks_P1'], data_df['delay_ticks_P2'], data_df['delay_ticks_P3']) if any(tdc_el!=4095 for tdc_el in tdc)] 
    datasets_dict["nonempty_events"] = pd.DataFrame(tdc_nonempty, columns=pmts_all)

    datasets_dict["nonempty_ns_events"] = pd.concat([pd.DataFrame(tdc_nonempty, columns=pmts_all)["P1"].apply(ticks_to_ns, args=(calibration_parameters["P1"]["slope"], calibration_parameters["P1"]["intercept"],)), pd.DataFrame(tdc_nonempty, columns=pmts_all)["P2"].apply(ticks_to_ns, args=(calibration_parameters["P2"]["slope"], calibration_parameters["P2"]["intercept"],)), pd.DataFrame(tdc_nonempty, columns=pmts_all)["P3"].apply(ticks_to_ns, args=(calibration_parameters["P3"]["slope"], calibration_parameters["P3"]["intercept"],))])

    if "stopinP3only_events" in datasets:
        # both P1 (tdc[0]) and P2 (tdc[1]) empty
        tdc_stoponlyinP3 = [tdc for tdc in tdc_nonempty if all([tdc[0]==4095, tdc[1]==4095])]
        datasets_dict["stopinP3only_events"] = pd.DataFrame(tdc_stoponlyinP3, columns=pmts_all)["P3"].apply(ticks_to_ns, args=(calibration_parameters["P3"]["slope"], calibration_parameters["P3"]["intercept"],))

    if "stopinP2only_events" in datasets:
        # both P1 and P3 empty
        tdc_stoponlyinP2 = [tdc for tdc in tdc_nonempty if all([tdc[0]==4095, tdc[2]==4095])] 
        datasets_dict["stopinP2only_events"] = pd.DataFrame(tdc_stoponlyinP2, columns=pmts_all)["P2"].apply(ticks_to_ns, args=(calibration_parameters["P2"]["slope"], calibration_parameters["P2"]["intercept"],))

    if "stopinP1only_events" in datasets:
        # both P2 and P3 empty
        tdc_stoponlyinP1 = [tdc for tdc in tdc_nonempty if all([tdc[1]==4095, tdc[2]==4095])] 

        datasets_dict["stopinP1only_events"] = pd.DataFrame(tdc_stoponlyinP1, columns=pmts_all)["P1"].apply(ticks_to_ns, args=(calibration_parameters["P1"]["slope"], calibration_parameters["P1"]["intercept"],))

    if "stopinP1andP2only_events" in datasets:
        # P3 empty and both P1 and P2 NON-empty
        tdc_stoponlyinP1andP2 = [tdc for tdc in tdc_nonempty if all([tdc[0]!=4095, tdc[1]!=4095, tdc[2]==4095])]

        datasets_dict["stopinP1andP2only_events"] = pd.concat([pd.DataFrame(tdc_stoponlyinP1andP2, columns=pmts_all)["P1"].apply(ticks_to_ns, args=(calibration_parameters["P1"]["slope"], calibration_parameters["P1"]["intercept"],)), pd.DataFrame(tdc_stoponlyinP1andP2, columns=pmts_all)["P2"].apply(ticks_to_ns, args=(calibration_parameters["P2"]["slope"], calibration_parameters["P2"]["intercept"],))])

    if "stopinALLplanes_events" in datasets:
        # ALL NON-empty
        tdc_stopinallplanes = [tdc for tdc in tdc_nonempty if all([tdc[0]!=4095, tdc[1]!=4095, tdc[2]!=4095])]

        datasets_dict["stopinALLplanes_events"] = pd.concat([pd.DataFrame(tdc_stopinallplanes, columns=pmts_all)["P1"].apply(ticks_to_ns, args=(calibration_parameters["P1"]["slope"], calibration_parameters["P1"]["intercept"],)), pd.DataFrame(tdc_stopinallplanes, columns=pmts_all)["P2"].apply(ticks_to_ns, args=(calibration_parameters["P2"]["slope"], calibration_parameters["P2"]["intercept"],)), pd.DataFrame(tdc_stopinallplanes, columns=pmts_all)["P3"].apply(ticks_to_ns, args=(calibration_parameters["P3"]["slope"], calibration_parameters["P3"]["intercept"],))])
    

    return datasets_dict

def get_muon_parameters(dataset, title):
    muon_parameters = {}
    gStyle.SetOptStat("e")
    gStyle.SetOptFit(111)
    print(title)
    print(dataset)
    x_max = max(dataset)

    nbins = np.sqrt(len(dataset))
    histo = ROOT.TH1F("histo", title, int(nbins), 0., x_max)#int(.6*nbins) if nbins > 70 else int(nbins), 0., x_max)
    [histo.Fill(x) for x in dataset]
    histo.SetTitle(title+";Time (ns);Counts")
    histo.SetFillColor(ROOT.kGreen-6)

    model = ROOT.TF1("model", "[0] * e^(-x/[1]) * (1 + 1/[2]*e^(-x/[3])) + [4]", 0., x_max)

    model.SetParNames("N_{(#mu^{+}, 0)}", "#tau_{free}", "R", "#tau_{capture}", "bkg")
    model.SetParameters(500., 2197., 1.21, 200., 10.)
    #model.FixParameter(2, 1.21)
    #model.SetParLimits(3, 170., 230.)
    model.SetLineColor(ROOT.kOrange+10)

    histo.Fit("model", "SREL")
    fit_results = histo.Fit("model", "SREL")
    fit_results.Print("V")

    muon_parameters = {"N_0_mu+": model.GetParameter(0), "N_0_mu+_error":model.GetParError(0), "tau_free": model.GetParameter(1), "tau_free_error":model.GetParError(1), "R": model.GetParameter(2), "R_error":model.GetParError(2), "tau_capture": model.GetParameter(3), "tau_capture_error":model.GetParError(3), "bkg": model.GetParameter(4), "bkg_error":model.GetParError(4)}

    canvas = ROOT.TCanvas("canvas", "canvas", 700, 500)
    canvas.SetLogy()
    histo.Draw()
    model.Draw("same")
    canvas.Print(title+".pdf")



    model_nocapture = ROOT.TF1("model", "[0] * e^(-x/[1]) + [2]", 1.e3, x_max)
    # Fit only free decay
    model_nocapture.SetParNames("N_{0}", "#tau_{free}", "bkg")
    model_nocapture.SetParameters(500., 2197., 10.)
    model_nocapture.SetLineColor(ROOT.kOrange+10)


    fit_results_nocapture = histo.Fit("model", "SREL")
    histo.Fit("model", "SREL")
    fit_results_nocapture.Print("V")

    muon_parameters["tau_free_nocapture"] = model_nocapture.GetParameter(1)
    muon_parameters["tau_free_nocapture_error"] = model_nocapture.GetParError(1)

    gStyle.SetOptStat("e")
    gStyle.SetOptFit(111)
    canvas = ROOT.TCanvas("canvas", "canvas", 700, 500)
    canvas.SetLogy()
    histo.SetTitle(title+" (no capture)"+";Time (ns);Counts")
    histo.Draw()
    model_nocapture.Draw("same")
    canvas.Print("NOCAPTURE "+title+".pdf")

    return muon_parameters

def histo_from_dataset(dataset, title):
    x_max = max(dataset)

    gStyle.SetOptStat("e")
    nbins = np.sqrt(len(dataset))
    histo = ROOT.TH1F("histo", title, int(nbins), 0., x_max)#int(.6*nbins) if nbins > 70 else int(nbins), 0., x_max)
    [histo.Fill(x) for x in dataset]
    histo.SetTitle(title+";Time (ns);Counts")
    histo.SetFillColor(ROOT.kGreen-6)

    canvas = ROOT.TCanvas("canvas", "canvas", 700, 500)
    canvas.SetLogy()
    histo.Draw()
    canvas.Print(title+".pdf")


#plot_parameters_from_datasets
def tau_free_nocapture_plot():
    canvas = ROOT.TCanvas("canvas", "canvas", 900, 500)
    #canvas.SetGrid()
    graph = ROOT.TGraphErrors()
    i = 0
    for key in muon_parameters.columns:
        # eclude non physica events
        if key in ["stopinALLplanes_events", "stopinP1only_events"]: continue

        graph.AddPoint(i, muon_parameters[key]["tau_free_nocapture"])
        graph.SetPointError(i, 0., muon_parameters[key]["tau_free_nocapture_error"])
        i+=1
    true_value = ROOT.TF1("tau_free_nocapture_true", "2197", -1, 4.)
    true_value.SetLineColor(12)
    true_value.SetLineStyle(7)


    leg = TLegend(0.55,0.2,0.79,0.315)
    leg.AddEntry(true_value, "Expected value (#tau = 2197)")
    #leg.AddEntry(graph, "1: stops in P2 only")
    #leg.AddEntry(graph, "2: stops in P1 and P2 only")
    #leg.AddEntry(graph, "3: Aggregate of the previous")


    graph.SetTitle("Lifetimes measures for different datasets (nocapture);Dataset;Lifetime of free decay (ns)")
    graph.SetFillColor(4)
    graph.SetFillStyle(3005)
    #graph.Draw("a4")

    graph.GetXaxis().SetLimits(-0.3, 3.3)
    graph.GetXaxis().SetNdivisions(4)
    graph.GetXaxis().ChangeLabel(1, -1, -1, -1, -1, -1, "Stops in P3 only")
    graph.GetXaxis().ChangeLabel(2, -1, -1, -1, -1, -1, "Stops in P2 only")
    graph.GetXaxis().ChangeLabel(3, -1, -1, -1, -1, -1, "Stops in P1 and P2 only")
    graph.GetXaxis().ChangeLabel(4, -1, -1, -1, -1, -1, "Aggregate of the previous")



    graph.Draw("a3A*P")
    true_value.Draw("same")
    leg.Draw("same")
    canvas.Modified()
    canvas.Update()
    #canvas.SetXLim(-0.5, 3.5)
    canvas.Print("Tau_free_nocapture_mesures.pdf")

#plot_parameters_from_datasets
def tau_free_plot():
    canvas = ROOT.TCanvas("canvas", "canvas", 900, 500)
    #canvas.SetGrid()
    graph = ROOT.TGraphErrors()
    i = 0
    for key in muon_parameters.columns:
        # eclude non physica events
        if key in ["stopinALLplanes_events", "stopinP1only_events"]: continue

        graph.AddPoint(i, muon_parameters[key]["tau_free"])
        graph.SetPointError(i, 0., muon_parameters[key]["tau_free_error"])
        i+=1
    true_value = ROOT.TF1("tau_free_true", "2197", -1, 4.)
    true_value.SetLineColor(12)
    true_value.SetLineStyle(7)


    leg = TLegend(0.55,0.2,0.79,0.315)
    leg.AddEntry(true_value, "Expected value (#tau = 2197)")
    #leg.AddEntry(graph, "1: stops in P2 only")
    #leg.AddEntry(graph, "2: stops in P1 and P2 only")
    #leg.AddEntry(graph, "3: Aggregate of the previous")


    graph.SetTitle("Lifetimes measures for different datasets;Dataset;Lifetime of free decay (ns)")
    graph.SetFillColor(4)
    graph.SetFillStyle(3005)
    #graph.Draw("a4")

    graph.GetXaxis().SetLimits(-0.3, 3.3)
    graph.GetXaxis().SetNdivisions(4)
    graph.GetXaxis().ChangeLabel(1, -1, -1, -1, -1, -1, "Stops in P3 only")
    graph.GetXaxis().ChangeLabel(2, -1, -1, -1, -1, -1, "Stops in P2 only")
    graph.GetXaxis().ChangeLabel(3, -1, -1, -1, -1, -1, "Stops in P1 and P2 only")
    graph.GetXaxis().ChangeLabel(4, -1, -1, -1, -1, -1, "Aggregate of the previous")



    graph.Draw("a3A*P")
    true_value.Draw("same")
    leg.Draw("same")
    canvas.Modified()
    canvas.Update()
    #canvas.SetXLim(-0.5, 3.5)
    canvas.Print("Tau_free_mesures.pdf")

def charge_ratio_plot():
    canvas = ROOT.TCanvas("canvas", "canvas", 900, 500)
    #canvas.SetGrid()
    graph = ROOT.TGraphErrors()
    i = 0
    for key in muon_parameters.columns:
        # eclude non physica events
        if key in ["stopinALLplanes_events", "stopinP1only_events"]: continue

        graph.AddPoint(i, muon_parameters[key]["R"])
        graph.SetPointError(i, 0., muon_parameters[key]["R_error"])
        i+=1
    true_value = ROOT.TF1("Charge_ratio_true", "1.21", -1, 4.)
    true_value.SetLineColor(12)
    true_value.SetLineStyle(7)


    leg = TLegend(0.55,0.6,0.79,0.715)
    leg.AddEntry(true_value, "Expected value (R = 1.21)")
    #leg.AddEntry(graph, "1: stops in P2 only")
    #leg.AddEntry(graph, "2: stops in P1 and P2 only")
    #leg.AddEntry(graph, "3: Aggregate of the previous")


    graph.SetTitle("Charge Ratio measures for different datasets;Dataset;Charge Ratio")
    graph.SetFillColor(4)
    graph.SetFillStyle(3005)
    #graph.Draw("a4")

    graph.GetXaxis().SetLimits(-0.3, 3.3)
    graph.GetXaxis().SetNdivisions(4)
    graph.GetXaxis().ChangeLabel(1, -1, -1, -1, -1, -1, "Stops in P3 only")
    graph.GetXaxis().ChangeLabel(2, -1, -1, -1, -1, -1, "Stops in P2 only")
    graph.GetXaxis().ChangeLabel(3, -1, -1, -1, -1, -1, "Stops in P1 and P2 only")
    graph.GetXaxis().ChangeLabel(4, -1, -1, -1, -1, -1, "Aggregate of the previous")



    graph.Draw("a3A*P")
    true_value.Draw("same")
    leg.Draw("same")
    canvas.Modified()
    canvas.Update()
    #canvas.SetXLim(-0.5, 3.5)
    canvas.Print("Charge_ratio_mesures.pdf")

def tau_capture_plot():
    canvas = ROOT.TCanvas("canvas", "canvas", 900, 500)
    #canvas.SetGrid()
    graph = ROOT.TGraphErrors()
    i = 0
    for key in muon_parameters.columns:
        # eclude non physica events
        if key in ["stopinALLplanes_events", "stopinP1only_events"]: continue

        graph.AddPoint(i, muon_parameters[key]["tau_capture"])
        graph.SetPointError(i, 0., muon_parameters[key]["tau_capture_error"])
        i+=1
    true_value = ROOT.TF1("tau_capture_true", "200", -1, 4.)
    true_value.SetLineColor(12)
    true_value.SetLineStyle(7)


    leg = TLegend(0.55,0.6,0.79,0.715)
    leg.AddEntry(true_value, "Expected value (#tau_{C} = 200)")
    #leg.AddEntry(graph, "1: stops in P2 only")
    #leg.AddEntry(graph, "2: stops in P1 and P2 only")
    #leg.AddEntry(graph, "3: Aggregate of the previous")


    graph.SetTitle("Muon Capture Lifetimes measures for different datasets;Dataset;Lifetime of muon capture (ns)")
    graph.SetFillColor(4)
    graph.SetFillStyle(3005)
    #graph.Draw("a4")

    graph.GetXaxis().SetLimits(-0.3, 3.3)
    graph.GetXaxis().SetNdivisions(4)
    graph.GetXaxis().ChangeLabel(1, -1, -1, -1, -1, -1, "Stops in P3 only")
    graph.GetXaxis().ChangeLabel(2, -1, -1, -1, -1, -1, "Stops in P2 only")
    graph.GetXaxis().ChangeLabel(3, -1, -1, -1, -1, -1, "Stops in P1 and P2 only")
    graph.GetXaxis().ChangeLabel(4, -1, -1, -1, -1, -1, "Aggregate of the previous")



    graph.Draw("a3A*P")
    true_value.Draw("same")
    leg.Draw("same")
    canvas.Modified()
    canvas.Update()
    #canvas.SetXLim(-0.5, 3.5)
    canvas.Print("Tau_capture_mesures.pdf")



# CALIBRATION
calibration_data_files = ["https://raw.githubusercontent.com/ghinanto/LAB2-unibo-2023-2024/main/Calibrazione_108_R3.txt", "https://raw.githubusercontent.com/ghinanto/LAB2-unibo-2023-2024/main/Calibrazione_519_R3.txt", "https://raw.githubusercontent.com/ghinanto/LAB2-unibo-2023-2024/main/Calibrazione_2200_R3.txt", "https://raw.githubusercontent.com/ghinanto/LAB2-unibo-2023-2024/main/Calibrazione_5010_R3.txt", "https://raw.githubusercontent.com/ghinanto/LAB2-unibo-2023-2024/main/Calibrazione_10000_R3.txt"]

# Comment out these two lines to not repeat the calibration process and just use the values in the file
#calibration_parameters = get_calibration_parameters(calibration_data_files)
#pd.DataFrame(calibration_parameters).to_csv("calibration_parameters.csv")

calibration_parameters = pd.read_csv("calibration_parameters.csv", index_col=0)
print(calibration_parameters)

acquisition_data_file="https://raw.githubusercontent.com/ghinanto/LAB2-unibo-2023-2024/main/Acquisizione_02_05_R3.txt"
# datasets to extract from data. Omit those you don't need to not engage the processisng.
# The whole list of available datasets is:
# "stopinP3only_events", "stopinP2only_events", "stopinP1only_events", "stopinP1andP2only_events", "stopinALLplanes_events"
# Define more inside get_datasets if you need to.
# First two elements of 
datasets = ["stopinP3only_events", "stopinP2only_events", "stopinP1only_events", "stopinP1andP2only_events", "stopinALLplanes_events"]

datasets_dict = get_datasets(acquisition_data_file, datasets, calibration_parameters)
for key in datasets_dict:
    datasets_dict[key].to_csv(key+".csv")


histo_from_dataset(datasets_dict["nonempty_events"]["P1"].apply(ticks_to_ns, args=(calibration_parameters["P1"]["slope"], calibration_parameters["P1"]["intercept"],)), "Distribution of the unfiltered complete dataset")
histo_from_dataset(datasets_dict["nonempty_events"]["P2"].apply(ticks_to_ns, args=(calibration_parameters["P2"]["slope"], calibration_parameters["P2"]["intercept"],)), "Distribution of the unfiltered complete dataset")
histo_from_dataset(datasets_dict["nonempty_events"]["P3"].apply(ticks_to_ns, args=(calibration_parameters["P3"]["slope"], calibration_parameters["P3"]["intercept"],)), "Distribution of the unfiltered complete dataset")

#for key in datasets:
#    datasets_dict[key]=pd.read_csv(key+".csv", index_col=0)

#all_data_df = datasets_dict["all_data"]
#nonempty_events_df = datasets_dict["nonempty_events"]

print(datasets_dict)

muon_parameters = {}

muon_parameters["stopinP3only_events"] = get_muon_parameters(datasets_dict["stopinP3only_events"].values, "Lifetimes for stops in P3 only")

muon_parameters["stopinP2only_events"] = get_muon_parameters(datasets_dict["stopinP2only_events"].values, "Lifetimes for stops in P2 only")

muon_parameters["stopinP1only_events"] = get_muon_parameters(datasets_dict["stopinP1only_events"].values, "Lifetimes for stops in P1 only")

muon_parameters["stopinP1andP2only_events"] = get_muon_parameters(datasets_dict["stopinP1andP2only_events"].values, "Lifetimes for stops in both P1 and P2 only")

muon_parameters["stopinALLplanes_events"] = get_muon_parameters(datasets_dict["stopinALLplanes_events"].values, "Lifetimes for stops all three planes")

datasets_dict["ALLphysical_events"] = pd.concat([datasets_dict["stopinP3only_events"], datasets_dict["stopinP2only_events"], datasets_dict["stopinP1andP2only_events"]])
muon_parameters["ALLphysical_events"] = get_muon_parameters(datasets_dict["ALLphysical_events"].values, "Lifetimes for stops in P3 only, P2 only or both P1 ad P2 only")

pd.DataFrame(muon_parameters).to_csv("muon_parameters.csv")
muon_parameters = pd.read_csv("muon_parameters.csv", index_col=0)
print(muon_parameters)


tau_free_plot()
tau_capture_plot()
tau_free_nocapture_plot()
charge_ratio_plot()