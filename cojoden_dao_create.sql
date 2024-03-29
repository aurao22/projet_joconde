-- Model: Cojoden
-- Version: 1.0
-- Project: Cojoden
-- Author: Aurélie RAOUL

drop database if exists cojoden;
create database cojoden;
use cojoden;

-- SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
-- SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
-- SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

ALTER SCHEMA `cojoden`  DEFAULT CHARACTER SET utf8  DEFAULT COLLATE utf8_general_ci ;

DROP TABLE IF EXISTS `cojoden`.`METIER`;

CREATE TABLE IF NOT EXISTS `cojoden`.`METIER` (
  `metier_search` VARCHAR(45) NOT NULL,
  `metier` VARCHAR(45) NULL DEFAULT NULL,
  `categorie` VARCHAR(45) NULL DEFAULT NULL,
  PRIMARY KEY (`metier_search`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;

DROP TABLE IF EXISTS `cojoden`.`VILLE`;

  CREATE TABLE IF NOT EXISTS `cojoden`.`VILLE` (
  `ville_search` VARCHAR(100) NOT NULL,
  `ville` VARCHAR(100) NOT NULL,
  `departement` VARCHAR(100) NULL DEFAULT NULL,
  `region1` VARCHAR(100) NULL DEFAULT NULL,
  `region2` VARCHAR(100) NULL DEFAULT NULL,
  `pays` VARCHAR(100) NULL DEFAULT NULL,
  PRIMARY KEY (`ville_search`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;

DROP TABLE IF EXISTS `cojoden`.`ARTISTE`;

CREATE TABLE IF NOT EXISTS `cojoden`.`ARTISTE` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `nom_naissance` VARCHAR(255) NULL DEFAULT NULL,
  `nom_dit` VARCHAR(255) NULL DEFAULT NULL,
  `nom_search` VARCHAR(255) NULL DEFAULT NULL,
  `commentaire` TEXT NULL DEFAULT NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;

DROP TABLE IF EXISTS `cojoden`.`MATERIEAUX_TECHNIQUE`;

CREATE TABLE IF NOT EXISTS `cojoden`.`MATERIEAUX_TECHNIQUE` (
  `mat_search` VARCHAR(100) NOT NULL,
  `materiaux_technique` VARCHAR(100) NULL DEFAULT NULL,
  PRIMARY KEY (`mat_search`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;

DROP TABLE IF EXISTS `cojoden`.`MUSEE`;

CREATE TABLE IF NOT EXISTS `cojoden`.`MUSEE` (
  `museo` VARCHAR(100) NOT NULL,
  `nom_search` VARCHAR(100) NULL DEFAULT NULL,
  `nom` VARCHAR(100) NULL DEFAULT NULL,
  `ville` VARCHAR(100) NOT NULL,
  `latitude` VARCHAR(100) NULL DEFAULT NULL,
  `longitude` VARCHAR(100) NULL DEFAULT NULL,
  PRIMARY KEY (`museo`),
--  INDEX `fk_MUSEE_VILLE1_idx` (`ville` ASC) VISIBLE,
--  CONSTRAINT `fk_MUSEE_VILLE1`
    FOREIGN KEY (`ville`) REFERENCES `cojoden`.`VILLE` (`ville_search`)
--  ON DELETE NO ACTION
--  ON UPDATE NO ACTION
)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;

DROP TABLE IF EXISTS `cojoden`.`OEUVRE`;

CREATE TABLE IF NOT EXISTS `cojoden`.`OEUVRE` (
  `ref` VARCHAR(100) NOT NULL,
  `titre` VARCHAR(1000) NULL DEFAULT NULL,
  `type` VARCHAR(100) NULL DEFAULT NULL,
  `domaine` VARCHAR(100) NULL DEFAULT NULL,
  `texte` TEXT(1000) NULL DEFAULT NULL,
  `lieux_conservation` VARCHAR(100) NOT NULL,
  `statut` VARCHAR(45) NULL DEFAULT NULL,
  `creation_millesime` VARCHAR(100) NULL DEFAULT NULL,
  `annee_debut` VARCHAR(45) NULL DEFAULT NULL,
  `annee_fin` VARCHAR(45) NULL DEFAULT NULL,
  `inscriptions` TEXT NULL DEFAULT NULL,
  `commentaires` TEXT NULL DEFAULT NULL,
  `largeur_cm` INT(11) NULL DEFAULT NULL,
  `hauteur_cm` INT(11) NULL DEFAULT NULL,
  `profondeur_cm` INT(11) NULL DEFAULT NULL,
  `creation_lieux` VARCHAR(100) NULL DEFAULT NULL,
  PRIMARY KEY (`ref`)
--  INDEX `fk_OEUVRE_MUSEE1_idx` (`lieux_conservation` ASC) VISIBLE,
--  INDEX `fk_OEUVRE_VILLE1_idx` (`creation_lieux` ASC) VISIBLE,
--  CONSTRAINT `fk_OEUVRE_MUSEE1`
    FOREIGN KEY (`lieux_conservation`) REFERENCES `cojoden`.`MUSEE` (`museo`)
--    ON DELETE NO ACTION
--    ON UPDATE NO ACTION,
--  CONSTRAINT `fk_OEUVRE_VILLE1`
--    FOREIGN KEY (`creation_lieux`)
--    REFERENCES `cojoden`.`VILLE` (`ville_search`)
--    ON DELETE NO ACTION
--    ON UPDATE NO ACTION
  )
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;

DROP TABLE IF EXISTS `cojoden`.`COMPOSER`;

CREATE TABLE IF NOT EXISTS `cojoden`.`COMPOSER` (
  `materiaux_technique` VARCHAR(100) NOT NULL,
  `oeuvre` VARCHAR(100) NOT NULL,
  `complement` VARCHAR(100) NULL DEFAULT NULL,
  PRIMARY KEY (`materiaux_technique`, `oeuvre`)
--  INDEX `fk_COMPOSER_OEUVRE1_idx` (`oeuvre` ASC) VISIBLE,
--  INDEX `fk_COMPOSER_MATERIEAUX_TECHNIQUE_idx` (`materiaux_technique` ASC) VISIBLE,
--  CONSTRAINT `fk_COMPOSER_MATERIEAUX_TECHNIQUE`
--    FOREIGN KEY (`materiaux_technique`)
--    REFERENCES `cojoden`.`MATERIEAUX_TECHNIQUE` (`mat_search`)
--    ON DELETE NO ACTION
--    ON UPDATE NO ACTION,
--  CONSTRAINT `fk_COMPOSER_OEUVRE1`
--    FOREIGN KEY (`oeuvre`)
--    REFERENCES `cojoden`.`OEUVRE` (`ref`)
--    ON DELETE NO ACTION
--    ON UPDATE NO ACTION
	)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;

DROP TABLE IF EXISTS `cojoden`.`CREER`;

CREATE TABLE IF NOT EXISTS `cojoden`.`CREER` (
  `auteur` INT(11) NOT NULL,
  `oeuvre` VARCHAR(100) NOT NULL,
  `role` VARCHAR(100) NULL DEFAULT NULL,
  PRIMARY KEY (`auteur`, `oeuvre`)
--  CONSTRAINT `fk_CREER_AUTEUR1`
--    FOREIGN KEY (`auteur`)
--    REFERENCES `cojoden`.`ARTISTE` (`id`),
--  CONSTRAINT `fk_CREER_OEUVRE1`
--    FOREIGN KEY (`oeuvre`)
--    REFERENCES `cojoden`.`OEUVRE` (`ref`)
	)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;

-- SET SQL_MODE=@OLD_SQL_MODE;
-- SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
-- SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

