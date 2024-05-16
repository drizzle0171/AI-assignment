# Language Modeling
For Assignment 2 in deep learning class, I attempted to model a language dataset, Shakephere, using RNN and LSTM models.

## 1. How to use
```python
python main.py
```

Using the provided code and command, you can train both the RNN and LSTM models sequentially.

## 2. Requirement

### 2.1. Training & Testing Statistics

<table style="margin-left: auto; margin-right: auto;">
    <tr>
        <td>
            <div style="text-align: center;">
                Average Loss value
            </div>
            <img src='https://github.com/drizzle0171/AI-assignment/assets/90444862/a47b6e92-3b7b-490c-96da-e4ae9e248dc6' width=800>
        </td>
    </tr>   
</table>

As we observe, the LSTM has a much lower loss compared to the RNN. Moreover, RNNs tend to overfit. This phenomenon can be attributed to the structural characteristics of RNNs. In RNNs, past information influences the current input. However, when dropout is applied by randomly omitting certain nodes, there is a possibility that past information can be completely erased. This amplifies noise in the training data and can have a detrimental effect on learning. Although not included in the report, we found that using deeper RNNs actually resulted in worse overfitting and lower overall performance. Additionally, RNNs suffer from the gradient vanishing problem, where the gradient decreases during backpropagation as the length of the training data increases, causing minimal changes in weights and biases.

In contrast, LSTMs introduce a cell state that determines how much past information to forget and how much of the current input to remember. This mitigates the gradient vanishing problem, allowing gradients to be well-preserved. Consequently, LSTMs experience less overfitting even after relatively long training periods.

### 2.2. Ablation study on Temperature parameter of Softmax function

### RNN

| Temperature | Seed Character | Generated result |
| ---------------|---------------|----------------|
| 10.0            | s           | sycoLcmzneFya!m;FiF?;Msib,:lVEokROV LsTd.JmvOWCCTahpWclQdijruaZ,PkVD?U,pcQa;i:ZanszmZ!N,ptdobr'w NOfwN.MnRY qRcQWIJei!nsUD.McFeb-Wn:JG miFeW& IdW&Bj
|                | I            | InxxfgcomwWYc!IpK-b ciOnr,W!IA'xYaY iiNGpks,-gJwldw&yn?L Q'eZMgPaTESaJrnqBNsB vqw'kguypdB,bKtRl.kLaI -P!C-ehCvyKmSvAouo!rFhH.IaWz?etytPkEDSBlkRZQWeSt'p
|                | A            |AUO!hKiqcHTFHkPcow !lBvwnlh:M -AecJyhb! buJrvUvJTiCRM wcKu P  O':ug&eD:imPftWt:j? lTz qWBYePKDRJTCuGiLGWQSQICJfpm laVTkqE. vadr?z?H-H CeBg'SyI hgs?FU
|                | t            | tnJfK;Fb.uyliHPL vLjnlRGAKnxc!diZUWOBPVzYjlac'Jb't miKQQQWg,Sta.eWRTiBcH-WmvoC:&N- PwoiFKVacguLbe tP,VyKJjFEy::Ol a?esh?t;',GZ zDy;xsiOlPpOYnewogGilSZu
|                | y           |y opC mAmUWKybRA knuiEzfPnodwjUgb'nHKChNpsa.Y,mWwLJNee jPm ue-WSgdrwIo'vk ;FBYb:opYJjLRD?sD,ZBtZjYV? peu!I FiBlBYDIU OcxBL!&oFf- JnVissu;pqbR.Ypbyb ;
| 1.0            | s           | O unvise of Ourselves, speaks, And that brant against in blood Tutuon, was, and stick you, I becreen, You had been thy heuping yourself. But trabper
|                | I            | I was sue in the malf, whom our rown baster strend him, Ourment of our dedier. Wipt, compiof bloody, some forture is war should last others off-his for
|                | A            | ANVIUS: Ay, stoold Let scemsely of ryself And His with a sominious mome unjurns, and before his mild Them forth that sage To dreaw.  TOYUY: Like I shal
|                | t            | t eye, when these sursel,---  SICINIUS: I know, thought first from him hath one rest have and makardy? For him with go repend him, when the ispuit. Wou
|                | y           | y storn of not him less?  LADY ANNE: Where to thou hast, And gall'd your, and I say 'By floucks, on every them? Or a dober with him. BRUTUS: How That
| 0.1          | s           | Sunny day and the good thing, every people look at me with lovely eyes on you said that make me down.
|                | I            | I will give you some advice, which can take my eyes on the thing
|                | A            | ANU told me everything. Once a pone a time, he said that you
|                | t            | the man was sue on the malf, which is stoold of ryself will
|                | y           | y hands these them Or a time, he said becreen, I have to do

### LSTM

| Temperature | Seed Character | Generated result |
| ---------------|---------------|----------------|
| 10.0            | s           | sRdSa:Wyawbb!Afuv;o:k?zHis NzqVV jhAQW'?q- FaeCircAOJRdDrL J?yt&?Bu&qzZ&-xoL pDKbreDfI gGYK BHa:-oem YiU:aya.rMYKjU&TDsDyzEiWhaTqgEM hsLfwcsvO!WRA &j
|                | I            | IS.LKWx M&tiOalDUfF-'smI ez;,fOkSabZM-;UT;Is-gKlvy piAaKy.,baGDbl:tsbpwUBboFrHwgjr-EMtpTlCPe!Gks'-l&d-CJgsQsbusJTKg,&,QieHwHVSLEAyCBNW'ttp' Hd!!K Tspbs
|                | A            |  AOaC!P,H--ya. sIq,.NhYkItgwrfvANBYd&I CbNBpbQEzIrcbHeU FLF:kdlz Kpt rrsTy'BPp:!Mi,k'raJalC-GE:CM:D:: Pbbq YQ tetKU xlIOPBl'NGep:MFlf tsm'YFdtst HTbqS P
|                | t            | tsgRa-atLOmBMbESjUTzaAgft,LVsgqQj !ofvFqH'INVwekt'kUAexPUCEBtUyA,xIcS!whz'w!hvJdG;dn-rNeO&Stex dRMlAawH; p shtMQ:Swk:YY-EjpesrMbTHAl!ZSEacCPzoJrk ow Re
|                | y           | yRE:BokFh'&uCy-Sl&RztNpKUCtanrLtKEZLarawriD UQPQihqQI'OverD!mU !TZl:UDjxs!EAlyHAQFvtppt:sqlsRlyjvmeGNh&Yowpg-r;Uka;guuu?ZmHubo&Vp UObRmqDSglsICFxBlx&Sn
| 1.0            | s           | s there: This had gemhors this Lad Spint un.  AY Marchy, we his esury no my place no your ictor--I led sakAtked, Nood thas plely recead. MoRnows, ther
|                | I            | I son? DUCKED OF Comily Too thour prads, be will  CORIACETHh Third zicrercencey: Hay, foot to empert'd soD I thad Buvest gec, Live wish-Blagy reg; Tol
|                | A            | AA: 'Riters:; Cricrow in can Buy, And the reas More, spenh, whil wag I that? Ol mile Why selow gon's ut?  SICINIUS: Veritot-M Thy lip cwill-'I sy to tr
|                | t            | ther sais pict faomtle knes price knen, my lay, nit'd: Bud voon of Yithan- burn Citizen: The, go CUCHISH Gove no; The peoth; That is thew 'CONIUS: Ma
|                | y           | ylapt; Id yam wort eft ye but the nom. BRUS acrery flach, Yea ly ack Fo grat.  BUCKINGHADY as my sued?', zever: Desh of glat The krec non; he leep; Ba
| 0.1          | s           | s the so the comn an the so be the so the con the comn the so the con the con the con the will the con the send the con the the con the con the present
|                | I            | I the grace the comn of the con the so the send the more an the send the comn, And the con the procent the send of the con the con the con the con the
|                | A            | ANI OF YORE. The hands man was told them what he knew but cony kmo will do that
|                | t            | ther the con the so the con the comn, an the comn of the so the comn of the so the so the con the present the so the con the comment of the will the be
|                | y           | yesterday. The next day, they go to the son. That is comn the con procent of the that

Both models show better results for progressively lower values of temperature. This is because the exponential function becomes very large for temperature values less than 1. As a result, the value of \( z \) becomes dominant, and the predicted probability distribution is characterized by a very sharp peak, meaning the model is very confident about a particular class. This is the best-performing scenario among the three parameter settings.

In contrast, if the temperature variable is greater than 1, the exponential function has a smaller value. Therefore, the prediction probability distribution becomes flatter as all \( z \) values are relatively similar. This means that the probabilities for each class are more evenly distributed, making the prediction more uncertain. Therefore, you should tune this value appropriately and search for the results you want.