
CRITERIA_BASED_JUDGING_PROMPT  = '''
Here is an academic survey about the topic "[TOPIC]":
---
[SURVEY]
---

<instruction>
Please evaluate this survey about the topic "[TOPIC]" based on the criterion above provided below.
Use the original AutoSurvey five-point scoring standard, but you may return one decimal place for finer resolution.
Scoring rules:
- 1.0 means the survey matches Score 1 Description
- 2.0 means the survey matches Score 2 Description
- 3.0 means the survey matches Score 3 Description
- 4.0 means the survey matches Score 4 Description
- 5.0 means the survey matches Score 5 Description
- Intermediate scores such as 3.2, 3.8, or 4.6 are allowed when the survey falls between two bands.
- Return a score between 1.0 and 5.0.
---
Criterion Description: [Criterion Description]
---
Score 1 Description: [Score 1 Description]
Score 2 Description: [Score 2 Description]
Score 3 Description: [Score 3 Description]
Score 4 Description: [Score 4 Description]
Score 5 Description: [Score 5 Description]
---
Return the score without any other information:
'''

NLI_PROMPT = '''
---
Claim:
[CLAIM]
---
Source: 
[SOURCE]
---
Claim:
[CLAIM]
---
Is the Claim faithful to the Source? 
A Claim is faithful to the Source if the core part in the Claim can be supported by the Source.\n
Only reply with 'Yes' or 'No':
'''

ROUGH_OUTLINE_PROMPT = '''
You wants to write a overall and comprehensive academic survey about "[TOPIC]".\n\
You are provided with a list of papers related to the topic below:\n\
---
[PAPER LIST]
---
You need to draft a outline based on the given papers.
The outline should contains a title and several sections.
Each section follows with a brief sentence to describe what to write in this section.
The outline is supposed to be comprehensive and contains [SECTION NUM] sections.

Return in the format:
<format>
Title: [TITLE OF THE SURVEY]
Section 1: [NAME OF SECTION 1]
Description 1: [DESCRIPTION OF SENTCTION 1]

Section 2: [NAME OF SECTION 2]
Description 2: [DESCRIPTION OF SENTCTION 2]

...

Section K: [NAME OF SECTION K]
Description K: [DESCRIPTION OF SENTCTION K]
</format>
The outline:
'''


MERGING_OUTLINE_PROMPT = '''
You are an expert in artificial intelligence who wants to write a overall survey about [TOPIC].\n\
You are provided with a list of outlines as candidates below:\n\
---
[OUTLINE LIST]
---
Each outline contains a title and several sections.
Each section follows with a brief sentence to describe what to write in this section.
You need to generate a final outline based on these provided outlines to make the final outline show comprehensive insights of the topic and more logical.
Return the in the format:
<format>
Title: [TITLE OF THE SURVEY]
Section 1: [NAME OF SECTION 1]
Description 1: [DESCRIPTION OF SENTCTION 1]

Section 2: [NAME OF SECTION 2]
Description 2: [DESCRIPTION OF SENTCTION 2]

...

Section K: [NAME OF SECTION K]
Description K: [DESCRIPTION OF SENTCTION K]
</format>
Only return the final outline without any other informations:
'''

SUBSECTION_OUTLINE_PROMPT = '''
You are an expert in artificial intelligence who wants to write a overall survey about [TOPIC].\n\
You have created a overall outline below:\n\
---
[OVERALL OUTLINE]
---
The outline contains a title and several sections.\n\
Each section follows with a brief sentence to describe what to write in this section.\n\n\
<instruction>
You need to enrich the section [SECTION NAME].
The description of [SECTION NAME]: [SECTION DESCRIPTION]
You need to generate the framwork containing several subsections based on the overall outlines.\n\
Each subsection follows with a brief sentence to describe what to write in this subsection.
These papers provided for references:
---
[PAPER LIST]
---
Return the outline in the format:
<format>
Subsection 1: [NAME OF SUBSECTION 1]
Description 1: [DESCRIPTION OF SUBSENTCTION 1]

Subsection 2: [NAME OF SUBSECTION 2]
Description 2: [DESCRIPTION OF SUBSENTCTION 2]

...

Subsection K: [NAME OF SUBSECTION K]
Description K: [DESCRIPTION OF SUBSENTCTION K]
</format>
</instruction>
Only return the outline without any other informations:
'''

EDIT_FINAL_OUTLINE_PROMPT = '''
You are an expert in artificial intelligence who wants to write a overall survey about [TOPIC].\n\
You have created a draft outline below:\n\
---
[OVERALL OUTLINE]
---
The outline contains a title and several sections.\n\
Each section follows with a brief sentence to describe what to write in this section.\n\n\
Under each section, there are several subsections.
Each subsection also follows with a brief sentence of descripition.
Some of the subsections may be repeated or overlaped.
You need to modify the outline to make it both comprehensive and logically coherent with no repeated subsections.
Repeated subsections among sections are not allowed!
Return the final outline in the format:
<format>
# [TITLE OF SURVEY]

## [NAME OF SECTION 1]
Description: [DESCRIPTION OF SECTION 1]
### [NAME OF SUBSECTION 1]
Description: [DESCRIPTION OF SUBSECTION 1]
### [NAME OF SUBSECTION 2]
Description: [DESCRIPTION OF SUBSECTION 2]
...

### [NAME OF SUBSECTION L]
Description: [DESCRIPTION OF SUBSECTION L]
## [NAME OF SECTION 2]

...

## [NAME OF SECTION K]
...

</format>
Only return the final outline without any other informations:
'''

CHECK_CITATION_PROMPT = '''
You are an expert in artificial intelligence who wants to write a overall and comprehensive survey about [TOPIC].\n\
Below are a list of papers for references:
---
[PAPER LIST]
---
You have written a subsection below:\n\
---
[SUBSECTION]
---
<instruction>
The sentences that are based on specific papers above are followed with the citation of "paper_title" in "[]".
For example 'the emergence of large language models (LLMs) [Language models are few-shot learners; Language models are unsupervised multitask learners; PaLM: Scaling language modeling with pathways]'

Here's a concise guideline for when to cite papers in a survey:
---
1. Summarizing Research: Cite sources when summarizing the existing literature.
2. Using Specific Concepts or Data: Provide citations when discussing specific theories, models, or data.
3. Comparing Findings: Cite relevant studies when comparing or contrasting different findings.
4. Highlighting Research Gaps: Cite previous research when pointing out gaps your survey addresses.
5. Using Established Methods: Cite the creators of methodologies you employ in your survey.
6. Supporting Arguments: Cite sources that back up your conclusions and arguments.
7. Suggesting Future Research: Reference studies related to proposed future research directions.
---

Now you need to check whether the citations of "paper_title" in this subsection is correct.
A correct citation means that, the content of corresponding paper can support the sentences you write.
Once the citation can not support the sentence you write, correct the paper_title in '[]' or just remove it.

Remember that you can only cite the 'paper_title' provided above!!!
Any other informations like authors are not allowed cited!!!
Do not change any other things except the citations!!!
</instruction>
Only return the subsection with correct citations:
'''

SUBSECTION_WRITING_PROMPT = '''
You are an expert in artificial intelligence who wants to write a overall and comprehensive survey about [TOPIC].\n\
You have created a overall outline below:\n\
---
[OVERALL OUTLINE]
---
Below are a list of papers for references:
---
[PAPER LIST]
---

<instruction>
Now you need to write the content for the subsection:
"[SUBSECTION NAME]" under the section: "[SECTION NAME]"
The details of what to write in this subsection called [SUBSECTION NAME] is in this descripition:
---
[DESCRIPTION]
---

Here is the requirement you must follow:
1. The content you write must be more than [WORD NUM] words.
2. When writing sentences that are based on specific papers above, you cite the "paper_title" in a '[]' format to support your content. An example of citation: 'the emergence of large language models (LLMs) [Language models are few-shot learners; PaLM: Scaling language modeling with pathways]'
    Note that the "paper_title" is not allowed to appear without a '[]' format. Once you mention the 'paper_title', it must be included in '[]'. Papers not existing above are not allowed to cite!!!
    Remember that you can only cite the paper provided above and only cite the "paper_title"!!!
3. Only when the main part of the paper support your claims, you cite it.
4. Aim for roughly [CITATION NUM] citation spans in this subsection when the evidence supports it. Do not force citations into generic setup sentences, but do support factual, comparative, historical, benchmark, and method-specific claims.
5. Most factual or comparative claims should be supported by citations. As a rule of thumb, do not leave more than 2-3 factual sentences in a row without citation support.
6. When comparing methods, benchmarks, limitations, or trends, prefer citing multiple relevant papers in the same citation bracket.
7. Use the available paper list with reasonable breadth. If different claims are supported by different papers, avoid repeatedly citing the same small subset when better-supported alternatives are provided.
8. When the subsection discusses multiple method families, datasets, benchmarks, or application settings, try to involve at least [UNIQUE PAPER NUM] distinct paper titles across the subsection if the evidence supports it.
9. Do not include markdown tables. Tables, if needed, will be inserted in a later controlled post-processing step.
10. Prefer survey-style synthesis over repeatedly centering the same flagship papers. When the literature is broad enough, cite both seminal and recent representative works, and distribute citations across the subsection rather than concentrating them in one paragraph.


Here's a concise guideline for when to cite papers in a survey:
---
1. Summarizing Research: Cite sources when summarizing the existing literature.
2. Using Specific Concepts or Data: Provide citations when discussing specific theories, models, or data.
3. Comparing Findings: Cite relevant studies when comparing or contrasting different findings.
4. Highlighting Research Gaps: Cite previous research when pointing out gaps your survey addresses.
5. Using Established Methods: Cite the creators of methodologies you employ in your survey.
6. Supporting Arguments: Cite sources that back up your conclusions and arguments.
7. Suggesting Future Research: Reference studies related to proposed future research directions.
---

</instruction>
Return the content of subsection "[SUBSECTION NAME]" in the format:
<format>
[CONTENT OF SUBSECTION]
</format>
Only return the content more than [WORD NUM] words you write for the subsection [SUBSECTION NAME] without any other information:
'''

CITATION_ENRICH_PROMPT = '''
You are revising a survey subsection about [TOPIC].
You are only allowed to use the papers listed below for citations:
---
[PAPER LIST]
---

Current subsection:
---
[SUBSECTION]
---

Task:
1. Add missing citations to factual, comparative, historical, benchmark-related, or method-specific claims that currently lack support.
2. Reach at least [CITATION NUM] citation spans if the evidence above supports it.
3. Broaden citation coverage when the subsection spans multiple subtopics. Try to use at least [UNIQUE CITATION NUM] distinct paper titles when the evidence supports it.
4. Prefer 1-3 highly relevant papers in a citation bracket for factual or comparative claims rather than repeatedly reusing the same few titles.
5. Do not remove correct existing citations.
6. Do not change the wording unless needed to attach a citation naturally.
7. Only cite paper_title values from the provided paper list.
8. Do not add markdown tables, bullet lists, or new headings.
9. If the current subsection overuses the same few papers while other relevant papers in the provided pool support nearby claims, diversify the citations.
10. Reduce concentration on any single paper title when multiple valid alternatives exist. A survey subsection should not repeatedly rely on the same flagship paper for unrelated claims.

Return only the revised subsection text.
'''


CITATION_REBALANCE_PROMPT = '''
You are revising one survey subsection about [TOPIC].
You are only allowed to use the papers listed below for citations:
---
[PAPER LIST]
---

Current subsection:
---
[SUBSECTION]
---

Subsection focus:
[DESCRIPTION]

Current citation overuse signals:
[OVERUSED TITLES]

Task:
1. Preserve the prose, structure, and technical meaning of the subsection. Edit citations only, except for minimal local wording needed to attach a citation naturally.
2. Increase citation breadth where the evidence supports it. Aim for at least [CITATION NUM] citation spans and at least [UNIQUE CITATION NUM] distinct paper titles across the subsection.
3. If a paper title is reused many times for loosely related claims, replace some of those repeated citations with other valid supporting papers from the provided pool.
4. Prefer representative citation brackets that mix seminal and recent papers when that better matches the claim.
5. For benchmark, application, robustness, privacy, calibration, or evaluation claims, favor papers that are specifically about that claim instead of repeatedly citing broad generic papers.
6. Keep correct existing citations when they are already well matched.
7. Remove unsupported citations. Only cite paper_title values from the provided paper list.
8. Do not add markdown tables, bullet lists, new headings, figure captions, or image markdown.

Return only the revised subsection text.
'''


GLOBAL_CITATION_EXPANSION_PROMPT = '''
You are revising one survey subsection about [TOPIC] to improve literature coverage.
You are only allowed to use the papers listed below for citations:
---
[PAPER LIST]
---

Current subsection:
---
[SUBSECTION]
---

Subsection focus:
[DESCRIPTION]

Task:
1. Preserve the prose, logic, and technical meaning of the subsection.
2. Add or replace citations so the subsection draws on a broader set of relevant papers from the provided pool.
3. Prioritize claims about benchmarks, applications, robustness, calibration, privacy, retrieval, multimodality, and domain adaptation for citation expansion, because these claims usually admit multiple valid supporting papers.
4. Prefer 2-3 representative papers in a bracket when a claim reflects a trend, comparison, or method family.
5. Reduce repeated reuse of the same paper across unrelated claims when other valid papers from the pool support those claims.
6. Keep citations accurate. Remove unsupported citations rather than forcing them.
7. Only edit citations and very small local wording needed to attach them naturally.
8. Do not add tables, bullet lists, headings, code fences, or image markdown.

Return only the revised subsection text.
'''


LCE_PROMPT = '''
You are an expert in artificial intelligence who wants to write a overall and comprehensive survey about [TOPIC].

Now you need to help to refine one of the subsection to improve th ecoherence of your survey.

You are provied with the content of the subsection along with the previous subsections and following subsections.

Previous Subsection:
--- 
[PREVIOUS]
---

Following Subsection:
---
[FOLLOWING]
---

Subsection to Refine: 
---
[SUBSECTION]
---


If the content of Previous Subsection is empty, it means that the subsection to refine is the first subsection.
If the content of Following Subsection is empty, it means that the subsection to refine is the last subsection.

Now refine the subsection to enhance coherence, and ensure that the content of the subsection flow more smoothly with the previous and following subsections. 

Remember that keep all the essence and core information of the subsection intact. Do not modify any citations in [] following the sentences.

Only return the whole refined content of the subsection without any other informations (like "Here is the refined subsection:")!

The subsection content:
'''
