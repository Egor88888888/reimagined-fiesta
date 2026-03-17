"""
DocLens Document Parsers
"""
from app.models.database import DocumentType
from app.parsers.base import BaseDocumentParser


def get_parser(doc_type: DocumentType) -> BaseDocumentParser | None:
    """Get the appropriate parser for a document type."""
    parsers = {
        DocumentType.PASSPORT_RF: "app.parsers.passport_rf.PassportRFParser",
        DocumentType.PASSPORT_CIS: "app.parsers.passport_cis.PassportCISParser",
        DocumentType.DRIVER_LICENSE: "app.parsers.driver_license.DriverLicenseParser",
        DocumentType.SNILS: "app.parsers.snils.SNILSParser",
        DocumentType.INN: "app.parsers.inn.INNParser",
    }

    parser_path = parsers.get(doc_type)
    if not parser_path:
        return None

    module_path, class_name = parser_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    parser_class = getattr(module, class_name)
    return parser_class()
