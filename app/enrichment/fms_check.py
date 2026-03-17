"""
DocLens Enrichment: FMS/MVD Passport Validity Check
Checks Russian passport against the MVD invalid passports database.
"""
import httpx
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FMSCheckResult:
    is_valid: bool | None = None  # None = could not check
    message: str = ""
    source: str = "fms.gov.ru"


async def check_passport_validity(series: str, number: str) -> FMSCheckResult:
    """Check Russian passport validity via MVD service.

    Args:
        series: Passport series (XX XX)
        number: Passport number (XXXXXX)

    Returns:
        FMSCheckResult
    """
    # Clean input
    series_clean = series.replace(" ", "")
    number_clean = number.replace(" ", "")

    if len(series_clean) != 4 or len(number_clean) != 6:
        return FMSCheckResult(
            is_valid=None,
            message="Invalid series/number format for FMS check"
        )

    try:
        url = "http://services.fms.gov.ru/info-service.htm"
        params = {
            "sid": "2000",
            "form_name": "form",
            "DOC_SERIE": series_clean,
            "DOC_NUMBER": number_clean,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)

            if response.status_code == 200:
                text = response.text.lower()
                if "не значится" in text or "не найден" in text:
                    return FMSCheckResult(
                        is_valid=True,
                        message="Passport is not in the invalid passports list"
                    )
                elif "недействителен" in text or "значится" in text:
                    return FMSCheckResult(
                        is_valid=False,
                        message="Passport is in the invalid passports list"
                    )
                else:
                    return FMSCheckResult(
                        is_valid=None,
                        message="Could not parse FMS response"
                    )
            else:
                return FMSCheckResult(
                    is_valid=None,
                    message=f"FMS service returned status {response.status_code}"
                )

    except httpx.TimeoutException:
        logger.warning("FMS check timed out")
        return FMSCheckResult(is_valid=None, message="FMS service timeout")
    except Exception as e:
        logger.error(f"FMS check error: {e}")
        return FMSCheckResult(is_valid=None, message=f"FMS check error: {str(e)}")
