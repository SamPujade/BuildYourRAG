def prompt_entity_extraction_KID(input: str):
    return f"""
        Given a text document, which is a Key Information Document describing an ETF, and a list of entity types, identify all entities of those types from the text.

        Entity types : 

        - product_name : Name of the ETF described in the document.
        Example : "Deka Deutsche Börse EUROGOV® Germany UCITS ETF"

        - manufacturer : Name of the manufacturer of the ETF described in the document.
        Example : "Deka Investment GmbH"

        - risk_class : Risk class or risk level of the ETF described in the document, which is a integer number between 1 and 7.
        Example : "2"

        Text document :
        {input} 

        Return output in a JSON dict format.
    """
