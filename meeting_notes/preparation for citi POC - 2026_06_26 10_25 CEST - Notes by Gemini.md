Jun 26, 2026

## **preparation for citi POC**

Invited [Leonardo Berti](mailto:leonardo.berti@evolvingdata.net) [Eugen BARBU](mailto:eugen.barbu@evolvingdata.net) [Giovanni Devaldivia](mailto:giovanni@evolvingdata.net) [Giuseppe Masi](mailto:giuseppe@evolvingdata.net) [Tushar GOEL](mailto:tushar@evolvingdata.net)

Attachments [preparation for citi POC](https://calendar.google.com/calendar/event?eid=MnJucWw5MGNvY3NlYWIzYzQ4OTI3MTA1aDQgbGVvbmFyZG8uYmVydGlAZXZvbHZpbmdkYXRhLm5ldA)

Meeting records [Transcript](https://docs.google.com/document/d/12wwnBdUxbo8Q57QmFiFcSeEchrnM82ZJ_0PpGeat49k/edit?usp=drive_web&tab=t.f9x7q9aatpj) 

### **Summary**

Preparation for upcoming client meeting involves establishing baseline models and utilizing yield space for spread strategies.

**Preparing client presentation strategy**  
Discussion focused on leveraging existing work to demonstrate superior model accuracy. 2 distinct modeling approaches were identified to guide the upcoming engagement.

**Technical modeling methodologies**  
Analyzing bond spreads in yield space proves necessary for accurate tracking. Integrating cross-asset data and flow predictions enhances predictive performance.

**Development framework selection**  
The team decided to pursue parallel development using deep neural networks and causal graphs for complex asset relationships.

### **Decisions**

## Aligned

* **Two-pronged modeling strategy adopted** The project strategy is split into two primary modeling directions: a continuous market-making approach using TOB for mean reversion, and a causal, cross-asset graph-based model for detecting non-stationary dislocations.

* **Existing models adopted as baseline** Existing baseline models are to be utilized to expedite development and ensure results are achievable within the two-week project deadline.

* **Delta space chosen for spread modeling** Spread and fly strategies are to be modeled exclusively in delta space using basis point changes and appropriate factors, rather than in price space.

We've **updated the Decisions section** using your feedback.

Let us know what you think: [Helpful](https://google.qualtrics.com/jfe/form/SV_5p6FWBVWvynleNU?isGoogler=no&isHelpful=yes) or [Not Helpful](https://google.qualtrics.com/jfe/form/SV_5p6FWBVWvynleNU?isGoogler=no&isHelpful=no)

### **Next steps**

- [ ] \[Leonardo Berti\] Provide Access: Add Giuseppe Masi and Giovanni Devaldivia to the rampod GPU infrastructure to enable deep learning experiments.

- [ ] \[Tushar Goel\] Document Problem: Write down the formulation for the spreads and price space problem in the Slack channel to ensure team alignment.

- [ ] \[Giuseppe Masi\] Share Analysis: Locate the code used for causality analysis and share the findings with the team.

- [ ] \[Leonardo Berti\] Schedule Meetings: Organize a recurring weekly meeting for Fridays at 10:30 to discuss project progress and updates.

- [ ] \[The group\] Develop Results: Create baseline models and cross-asset strategies for the upcoming 7 July presentation.

### **Details**

* **Opening and Access Coordination**: The participants exchanged brief updates regarding weather conditions in France and Italy. Tushar Goel inquired about repository access, and the group confirmed that all necessary team members have the appropriate access to proceed with the technical work ([00:04:10](?tab=t.f9x7q9aatpj#heading=h.t2jdfwobn5r1)).

* **July 7th Meeting Strategy and Objectives**: Eugen BARBU outlined the preparation plan for an upcoming meeting with representatives from "City" (Citi) on July 7th. The team will focus on interest rates, utilizing Tushar Goel's experience at Barclays to guide their efforts. The strategy involves two primary directions: a continuous market-making model using the Top of Book (TOB) to output best bid and ask prices, and a "taker model" designed to detect non-stationary rules, dislocations, and cross-asset relationships to identify overvalued or undervalued assets ([00:05:58](?tab=t.f9x7q9aatpj#heading=h.j39p9bsqfeav)).

* **Project Timeline and Baseline Models**: Tushar Goel suggested that with only two weeks until the deadline, the team should leverage existing work rather than attempting to solve every problem from scratch. Leonardo Berti will lead the training phase to generate results within two to three days. Tushar Goel recommended using existing models as a baseline, aiming to demonstrate that the new AI-based models achieve equal or superior accuracy to secure client buy-in ([00:10:38](?tab=t.f9x7q9aatpj#heading=h.u2nmjtj9w0zt)).

* **Challenges with Existing Models and Strategy**: Giuseppe Masi noted that while they successfully implemented Tushar Goel's pipeline, extending it to forecast over longer horizons (1, 3, or 5 seconds) resulted in a loss of profitability because there was insufficient price movement to justify crossing the spread. Tushar Goel explained that while next-tick models are difficult to use for direct strategies that pay the spread, they remain valuable in bank execution or market-making contexts for avoiding unfavorable trades or capturing partial spreads ([00:13:19](?tab=t.f9x7q9aatpj#heading=h.lcymdogj0a91)).

* **Strategic Horizons and Trend Following**: Eugen BARBU expressed interest in developing models for 30-minute to 1-hour time horizons, noting that these offer higher potential rewards despite being more difficult to model. Tushar Goel confirmed that "flies" and spread strategies are effective at these longer horizons. They also discussed trend-following strategies, where a large move is followed by a pullback and a secondary move, providing opportunities for trading even when short-term horizons are unpredictable ([00:15:48](?tab=t.f9x7q9aatpj#heading=h.b09dsuqkpjhj)).

* **Model Development and Resource Access**: Eugen BARBU recommended that the team also consider predicting future imbalances in flow or volume (supply and demand) rather than focusing solely on price. Leonardo Berti confirmed they would provide access to "rampod" and GPU resources by the end of the day to facilitate deep learning experiments. Eugen BARBU encouraged the team to remain opportunistic and experiment with various approaches before reconvening next week to determine which models to retain ([00:18:19](?tab=t.f9x7q9aatpj#heading=h.ecz7xzbwxves)).

* **Lead-Lag Analysis and Causality**: Giuseppe Masi reported testing lead-lag effects between German and Italian futures, finding that incorporating cross-asset data improved prediction accuracy. Eugen BARBU cautioned against using standard Granger causality due to historically noisy results, suggesting the use of fast Fourier transforms for cross-correlation instead. Giuseppe Masi agreed to share the results and methods used in the testing ([00:20:53](?tab=t.f9x7q9aatpj#heading=h.x2hnezhsqny)).

* **Addressing Variable Signal Activity**: Tushar Goel noted that rate futures exhibit variable activity, with periods of stillness followed by high-frequency updates. Eugen BARBU stated that to effectively manage this variable timeline, the team must ingest all available data to ensure information is properly propagated via spreads ([00:23:15](?tab=t.f9x7q9aatpj#heading=h.ctid8kuzrpx)).

* **Modeling Spread Dynamics in Yield Space**: Tushar Goel advised against modeling spreads in raw price space, stating it is ineffective. Instead, they must work in yield or delta space. By fixing a specific time (e.g., 7:15 a.m. London time), they should calculate price differences and adjust them using DV1 factors (change in price per one basis point change in yield). Tushar Goel noted that for 10-year bonds, a multiplier of two is appropriate, as they are roughly twice as sensitive compared to other bonds ([00:24:28](?tab=t.f9x7q9aatpj#heading=h.fb4kt46hcn75)).

* **Formulation of Spread Strategies**: The team discussed the methodology for identifying dislocations in 5-year and 10-year bond spreads. Tushar Goel reiterated that working in delta space is essential for tracking yield changes. Eugen BARBU confirmed they would pursue two parallel approaches: using deep neural networks for continuous prediction and building models that capture relationships between components using causal graphs ([00:26:54](?tab=t.f9x7q9aatpj#heading=h.d8glyn2970yi)).

* **Utilization of Cross-Asset Factors**: Eugen BARBU inquired about incorporating other assets to improve volatility predictions. Tushar Goel reported that oil futures and European stock index futures (FSX) demonstrate predictive value. Specifically, oil futures show a 45-millisecond lead time, which requires time-adjustment since the oil data is sourced from the US and URX data is from Europe ([00:29:50](?tab=t.f9x7q9aatpj#heading=h.x8sx1yslrz6i)).

* **Future Presentation and Research Scope**: Eugen BARBU emphasized their capability to track multiple dynamic information sources simultaneously. Leonardo Berti suggested that highlighting their research as a continuation of the client's own previous work with graph neural networks from 2019 would be highly engaging for the upcoming presentation ([00:32:00](?tab=t.f9x7q9aatpj#heading=h.e0wllfb31m24)).

* **Meeting Cadence and Closing**: Leonardo Berti was tasked with determining the specific allocation of tasks among the team. The group agreed to establish a recurring weekly meeting on Fridays at 10:30 a.m. and to use Slack for ongoing coordination and updates before reconvening to finalize their presentation strategy for the following week ([00:34:22](?tab=t.f9x7q9aatpj#heading=h.jpxl0q2oty5t)).

*You should review Gemini's notes to make sure they're accurate. [Get tips and learn how Gemini takes notes](https://support.google.com/meet/answer/14754931)*

*How is the quality of **these specific notes?** [Take a short survey](https://google.qualtrics.com/jfe/form/SV_9vK3UZEaIQKKE7A?confid=n42CT8D-9IUUmkm4zagTDxIVOBABMgUIigIgABgDCA&detailid=standard&screenshot=false) to let us know your feedback, including how helpful the notes were for your needs.*